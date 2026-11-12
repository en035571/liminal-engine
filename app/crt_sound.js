// CRT SOUND (11-05): the picture chain's twin, run as one AudioWorklet so the live preview and the offline export
// are the same code. Order follows the picture: TAPE (deck) → BROADCAST (RF) → SET (speaker, flyback, power) → lo-fi.
// Everything that comes from a picture slider is scaled by `amt` ("Sound follows picture"); speaker / whine / buzz /
// crush / vol are the Sound group's own sliders. Tape dropouts use the SAME hash as the TAPEMECH shader, so a crackle
// lands on the field the picture drops out on.
//
// Time: `ptime` is the picture's clock. Offline it is (sample time − LAT) because the whole render is trimmed by LAT
// afterwards (LAT = the wow/flutter delay line's base delay), so picture and sound line up to the sample. Live it is
// the page's performance clock, sent with every frame.

const LAT = 0.008, NY = 486, FIELD = 59.94;

// ---- the shader's hash, in 32-bit integer JS (COMMON.pcg / h3, TAPEMECH.hh) ----
function pcg(v){ const st=(Math.imul(v>>>0,747796405)+2891336453)>>>0; const w=Math.imul(((st>>>((st>>>28)+4))^st)>>>0,277803737)>>>0; return ((w>>>22)^w)>>>0; }
function h3(a,b,c){ return pcg(((a|0)>>>0) + pcg(((b|0)>>>0) + pcg((c|0)>>>0)) >>> 0)/4294967295; }
function hh(a,b,c){ return h3(a*7+13,b,c); }
const smooth=(x,a,b)=>{ const t=Math.min(Math.max((x-a)/(b-a),0),1); return t*t*(3-2*t); };

// ---- RBJ biquads ----
class BQ { constructor(){ this.b0=1; this.b1=0; this.b2=0; this.a1=0; this.a2=0; this.z=[[0,0],[0,0]]; }
  set(type, f, q, gainDb, sr){ f=Math.min(Math.max(f,10),sr*.45); const w=2*Math.PI*f/sr, c=Math.cos(w), s=Math.sin(w), al=s/(2*q);
    let b0,b1,b2,a0,a1,a2;
    if(type==='lp'){ b0=(1-c)/2; b1=1-c; b2=(1-c)/2; a0=1+al; a1=-2*c; a2=1-al; }
    else if(type==='hp'){ b0=(1+c)/2; b1=-(1+c); b2=(1+c)/2; a0=1+al; a1=-2*c; a2=1-al; }
    else if(type==='bp'){ b0=al; b1=0; b2=-al; a0=1+al; a1=-2*c; a2=1-al; }
    else { const A=Math.pow(10,gainDb/40); b0=1+al*A; b1=-2*c; b2=1-al*A; a0=1+al/A; a1=-2*c; a2=1-al/A; }
    this.b0=b0/a0; this.b1=b1/a0; this.b2=b2/a0; this.a1=a1/a0; this.a2=a2/a0; }
  run(x, ch){ const z=this.z[ch]; const y=this.b0*x+z[0]; z[0]=this.b1*x-this.a1*y+z[1]; z[1]=this.b2*x-this.a2*y; return y; } }

// tape formats: [bandwidth Hz, flutter %, hiss]  (off, Betacam SP, U-matic, VHS SP, VHS EP)
const TAPE = [null, [15000,.02,.0015], [12000,.08,.004], [10000,.2,.007], [6000,.35,.012]];
const TIMED = ['snow','signal','roll','jitter','warm','off','tone','burst','dropMul','deckMute','crease'];

class CRTSound extends AudioWorkletProcessor {
  constructor(opt){
    super();
    const o=(opt&&opt.processorOptions)||{};
    this.p=Object.assign({amt:.6, speaker:.35, whine:.15, buzz:.1, crush:0, vol:1, snow:0, signal:1, ghost:0, jitter:0, hum:0, roll:0, interf:0,
      power:1, warm:1, off:1, tone:0, burst:0, dropMul:1, deckMute:0, crease:0, tapeFmt:0, tapeGen:1, tapeWave:1, tracking:0, dropout:0, lines:480, t:0}, o.p||{});
    this.sched=o.sched||null;          // offline: {key: Float32Array per field}
    this.tOff=0;                       // live: picture clock − audio clock
    this.port.onmessage=e=>{ if(e.data&&e.data.p){ Object.assign(this.p,e.data.p); if('t' in e.data.p) this.tOff=e.data.p.t-currentTime; } };
    this.n=0;
    const N=Math.ceil(sampleRate*.5); this.buf=[new Float32Array(N),new Float32Array(N)]; this.post=[new Float32Array(N),new Float32Array(N)]; this.N=N; this.w=0;
    this.tapeLP=new BQ(); this.spHP=new BQ(); this.spLP=new BQ(); this.spPk=new BQ(); this.stat=new BQ(); this.hissHP=new BQ();
    this.hissHP.set('hp',2500,.7,0,sampleRate);
    this.jit=0; this.jitT=0; this.field=-1; this.drops=[]; this.dEnv=0;
    this.prevWarm=1; this.prevOff=1; this.degT=-1; this.popT=-1;
    this.hold=[0,0]; this.holdAcc=0; this.gProg=1; this.key='';
    this.seed=12345;
  }
  rnd(){ this.seed=(Math.imul(this.seed,1664525)+1013904223)>>>0; return this.seed/2147483648-1; }
  // the dropouts the TAPEMECH shader draws in field `f`, as audible events (start in samples from the field start)
  dropsFor(f, P){
    const out=[]; if(P.tapeFmt<1 || P.tapeFmt>4 || P.amt<=0) return out;   // digital tape has no oxide streaks to hear
    const e=.06*NY, fs=sampleRate/FIELD;
    for(let fl=0; fl<NY; fl++){
      // same test as the shader; the tracking term (edge lines) is left out: tracking has its own buzz below
      const pd=P.dropout/1000*(P.dropMul||1); if(pd<=0) break;
      if(hh(fl,f,3)<pd){ const h2=hh(fl,f,6), len=h2*h2*.5;
        if(len>=.45) out.push({s:fl/NY*fs, d:sampleRate*(.004+len*.03), a:Math.min(1,len*1.6)}); }   // only the long streaks are heard
    }
    return out;
  }
  process(inputs, outputs){
    const inp=inputs[0]||[], out=outputs[0], L=out[0], R=out[1]||out[0], n=L.length, sr=sampleRate;
    const P=this.p;
    // offline: the timed values come from the schedule at this block's picture time
    const t0=this.sched ? (this.n/sr - LAT) : (currentTime + this.tOff);
    if(this.sched){ const i=Math.max(0, Math.floor(t0*FIELD)); for(const k of TIMED){ const a=this.sched[k]; if(a) P[k]=a[Math.min(i,a.length-1)]; } }
    const amt=P.amt, F=Math.round(P.tapeFmt), tf=F>0?TAPE[F]:null, gen=Math.max(1,Math.round(P.tapeGen));
    // filters, recomputed only when their inputs move
    const key=[F,gen,P.speaker,P.roll>0.001?Math.round(t0*20):0].join(',');
    if(key!==this.key){ this.key=key;
      if(tf) this.tapeLP.set('lp', tf[0]*Math.pow(.82,gen-1)/(1+.25*Math.max(0,P.tracking)/100), .707, 0, sr);
      const s=P.speaker; this.spHP.set('hp', 40+s*340, .707, 0, sr); this.spLP.set('lp', 19000-s*14000, .8, 0, sr); this.spPk.set('pk', 1150, .9, s*7, sr);
      this.stat.set('bp', 500+2600*(.5+.5*Math.sin(t0*5.7)), 2.5, 0, sr); }
    // edges of the switch-on / switch-off sequence
    if(this.prevWarm<.02 && P.warm>=.02) this.degT=0;
    if(this.prevOff>=.995 && P.off<.995) this.popT=0;
    this.prevWarm=P.warm; this.prevOff=P.off;
    const flutter=tf ? (tf[1]+P.tapeWave*.06+P.tracking*.004)*Math.pow(1.25,gen-1) : 0;
    const wowA=amt*(flutter + (P.crease||0)*1.2)/100/(2*Math.PI*.55),   // a crease drags the tape: the sound warbles
          flA=amt*.6*flutter/100/(2*Math.PI*6.1), jitA=amt*P.jitter*.0006/(2*Math.PI*5);
    const drive=1+amt*((F>=5?0:F>=3?.6:F>0?.25:0)+(F>=5?0:.35*(gen-1))), dN=Math.tanh(drive);   // digital copies don't saturate
    const hiss=tf? amt*tf[2]*(1+.6*(gen-1)) : 0;
    const sigLv=Math.max(0,Math.min(1.5,P.signal));
    const progG=Math.min(1,sigLv)*(1+Math.max(0,sigLv-1)*.8) * P.warm * smooth(P.off,.45,.95) * P.power * (1-(P.deckMute||0));   // a VCR mutes in pause and search
    const powerOn=(P.power*P.warm*P.off)>.05?1:0;
    const staticLv=amt*powerOn*(P.snow*.22 + Math.max(0,1-sigLv)*.1), sweepLv=amt*powerOn*P.roll*.18;
    const humLv=amt*P.hum*.3*powerOn, buzzLv=(P.buzz*.035 + amt*P.roll*.05)*powerOn, interfLv=amt*P.interf*.08*powerOn;
    const ghostG=amt*Math.min(.6,P.ghost*1.4), whineLv=P.whine*.012*powerOn*P.warm, whineF=15734*(P.lines/480)*(.35+.65*P.off);
    const trkBuzz=tf? amt*P.tracking/100*.12 : 0;
    const crush=P.crush, holdN=1+crush*crush*9, q=Math.pow(2, 16-crush*12-1);
    const fs=sr/FIELD, ch2=inp.length>1;
    for(let i=0;i<n;i++){
      const t=t0+i/sr;
      // a new field: collect the dropouts the picture draws in it
      const f=Math.floor(t*FIELD);
      const abs=this.n+i;
      if(f!==this.field){ this.field=f; const base=abs+(f/FIELD - t)*sr;
        this.drops=this.drops.filter(d=>d.s+d.d>abs).concat(this.dropsFor(f,P).map(d=>({s:base+d.s, d:d.d, a:d.a}))); }
      let dTarget=Math.min(1,(P.burst||0)*.95); for(const d of this.drops){ if(abs>=d.s && abs<d.s+d.d){ dTarget=Math.max(dTarget,d.a); } }
      this.dEnv += (dTarget-this.dEnv)*(dTarget>this.dEnv?.25:.004);
      // jitter: smoothed random walk, a few hertz
      if(--this.jitT<=0){ this.jitT=sr/(3+Math.random()*6); this.jitN=this.rnd(); } this.jit+=(this.jitN-this.jit)*.0004;
      const xl=inp[0]?inp[0][i]:0, xr=ch2?inp[1][i]:xl;
      // DECK: wow / flutter / sync wobble through one modulated delay
      const w=this.w; this.buf[0][w]=xl; this.buf[1][w]=xr;
      const dly=(LAT + wowA*Math.sin(2*Math.PI*.55*t) + flA*Math.sin(2*Math.PI*6.1*t+1.3) + jitA*this.jit)*sr;
      const rp=w-dly, i0=Math.floor(rp), fr=rp-i0, a0=(i0%this.N+this.N)%this.N, a1=(a0+1)%this.N;
      let yl=this.buf[0][a0]*(1-fr)+this.buf[0][a1]*fr, yr=this.buf[1][a0]*(1-fr)+this.buf[1][a1]*fr;
      this.gProg+=(progG-this.gProg)*.002; yl*=this.gProg; yr*=this.gProg;
      if(tf){ yl=Math.tanh(yl*drive)/dN; yr=Math.tanh(yr*drive)/dN; yl=this.tapeLP.run(yl,0); yr=this.tapeLP.run(yr,1);
        const h=this.hissHP.run(this.rnd(),0)*hiss*powerOn; yl+=h; yr+=h*.9;
        const saw=((t*FIELD)%1); const tb=trkBuzz*powerOn*this.rnd()*(saw<.18?1:.15); yl+=tb; yr+=tb; }
      // dropout: the sound dips and crackles with the picture
      const dig=F>=5, dip=1-amt*(dig?1:.85)*this.dEnv; const cr=dig?0:amt*.25*this.dEnv*this.rnd()*powerOn;   // digital drops out to silence, no crackle
      yl=yl*dip+cr; yr=yr*dip+cr;
      // RF: ghost = multipath echo, static, sweep while rolling, hum, sync buzz, interference
      this.post[0][w]=yl; this.post[1][w]=yr;
      if(ghostG>0){ const g1=(w-Math.round(.045*sr)+this.N)%this.N, g2=(w-Math.round(.11*sr)+this.N)%this.N;
        yl+=ghostG*(this.post[0][g1]+.5*this.post[0][g2]); yr+=ghostG*(this.post[1][g1]+.5*this.post[1][g2]); }
      this.w=(w+1)%this.N;
      const nz=this.rnd(); let add=nz*staticLv + this.stat.run(nz,1)*sweepLv*4;
      add+=humLv*(Math.sin(2*Math.PI*59.94*t)+.5*Math.sin(2*Math.PI*119.88*t)+.25*Math.sin(2*Math.PI*179.82*t));
      add+=buzzLv*Math.tanh(6*Math.sin(2*Math.PI*59.94*t));
      if(interfLv>0){ const v=Math.sin(Math.floor(t*3)*91)*43; if(v-Math.floor(v)>=.6) add+=interfLv*Math.sin(2*Math.PI*1800*t)*(.5+.5*Math.sin(2*Math.PI*6.4*t)); }
      add+=P.tone*.12*Math.sin(2*Math.PI*1000*t);
      yl+=add; yr+=add;
      // SET: the TV's own speaker (mono-ish, band-limited, a presence bump), flyback whine, degauss, power pop
      const m=(yl+yr)/2, s=P.speaker; yl+= (m-yl)*s; yr+=(m-yr)*s;
      if(s>0.001){ yl=this.spPk.run(this.spLP.run(this.spHP.run(yl,0),0),0); yr=this.spPk.run(this.spLP.run(this.spHP.run(yr,1),1),1);
        const sd=1+s*1.2; yl=Math.tanh(yl*sd)/sd*(1+s*.15); yr=Math.tanh(yr*sd)/sd*(1+s*.15); }
      const wh=whineLv*Math.sin(2*Math.PI*whineF*t); yl+=wh; yr+=wh;
      if(this.degT>=0){ const dt=this.degT/sr, e=Math.exp(-dt/.35), lv=.3*(.4+.6*amt);
        const dg=lv*(e*Math.tanh(3*(Math.sin(2*Math.PI*59.94*dt)*.6+Math.sin(2*Math.PI*119.88*dt)*.4)) + 1.6*Math.exp(-dt/.08)*Math.sin(2*Math.PI*42*dt));
        yl+=dg; yr+=dg; if(++this.degT>sr*2) this.degT=-1; }
      if(this.popT>=0){ const dt=this.popT/sr, pp=.35*(.4+.6*amt)*(Math.exp(-dt/.05)*Math.sin(2*Math.PI*55*dt)+ (dt<.002?this.rnd()*.6:0));
        yl+=pp; yr+=pp; if(++this.popT>sr*.5) this.popT=-1; }
      // LO-FI: sample-rate hold + bit depth
      if(crush>0.001){ this.holdAcc+=1; if(this.holdAcc>=holdN){ this.holdAcc-=holdN; this.hold[0]=Math.round(yl*q)/q; this.hold[1]=Math.round(yr*q)/q; } yl=this.hold[0]; yr=this.hold[1]; }
      yl*=P.vol; yr*=P.vol;
      // soft ceiling, transparent below 0.8
      const lim=v=>{ const a=Math.abs(v); return a<.8?v:Math.sign(v)*(.8+.2*Math.tanh((a-.8)/.2)); };
      L[i]=lim(yl); if(R!==L) R[i]=lim(yr);
    }
    this.n+=n;
    return true;
  }
}
registerProcessor('crt-sound', CRTSound);
