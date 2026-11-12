// "Update available" bar for shared copies (Eric's own copy never shows it: the server reports enabled:false).
// Asks first, never updates mid-render (the server refuses), shows progress, then the app reopens itself.
(() => {
  const css = `
#upd{position:fixed;right:16px;bottom:16px;z-index:9999;width:min(520px,calc(100vw - 32px));
  background:var(--panel,#141418);border:1px solid var(--line2,#30303a);border-radius:var(--r,4px);
  font:12px/1.5 var(--mono,"SF Mono",Menlo,monospace);color:var(--ink,#e6e4e0);box-shadow:0 8px 28px rgba(0,0,0,.45)}
#upd[hidden]{display:none}
#upd .r{display:grid;grid-template-columns:8px minmax(0,1fr) auto;gap:10px;align-items:center;padding:10px 12px}
#upd .dot{width:8px;height:8px;border-radius:50%;background:var(--acc,#5f8dff)}
#upd .t{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#upd .t b{font-weight:400;color:var(--ink,#e6e4e0)} #upd .t span{color:var(--mute,#8e8a86)}
#upd .bs{display:flex;gap:6px}
#upd button{font:inherit;font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;border-radius:var(--r,4px);
  padding:5px 10px;border:1px solid var(--line2,#30303a);background:transparent;color:var(--mute,#8e8a86)}
#upd button:hover{color:var(--ink,#e6e4e0)}
#upd button.go{background:var(--acc,#5f8dff);border-color:var(--acc,#5f8dff);color:var(--accink,#07101f)}
#upd .notes{border-top:1px solid var(--line,#232329);padding:10px 12px 12px 30px;color:var(--mute,#8e8a86);
  white-space:pre-wrap;max-height:40vh;overflow:auto;font-family:var(--sans,system-ui,sans-serif);font-size:13px}
#upd .bar{height:2px;background:var(--line,#232329)} #upd .bar i{display:block;height:100%;width:0;background:var(--acc,#5f8dff);transition:width .3s}
#upd.err .dot{background:var(--bad,#e05252)}`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);
  const el = document.createElement("div"); el.id = "upd"; el.hidden = true;
  el.innerHTML = `<div class="r"><i class="dot"></i><div class="t"></div><div class="bs"></div></div>
    <div class="bar" hidden><i></i></div><div class="notes" hidden></div>`;
  document.body.appendChild(el);
  const $ = s => el.querySelector(s);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const later = () => { try { sessionStorage.setItem("upd-later", "1"); } catch (e) {} el.hidden = true; };
  const isLater = () => { try { return sessionStorage.getItem("upd-later") === "1"; } catch (e) { return false; } };
  let polling = null;

  const post = p => fetch(p, {method: "POST"}).then(r => r.json());
  function show(s) {
    const busy = ["downloading", "installing", "packages", "restarting"].includes(s.stage);
    el.classList.toggle("err", s.stage === "error");
    $(".bar").hidden = !busy; $(".bar i").style.width = (s.pct || 0) + "%";
    const t = $(".t"), bs = $(".bs");
    if (busy) {
      const what = {downloading: "Downloading", installing: "Installing", packages: "Installing new packages (a minute or two)",
                    restarting: "Updated · reopening"}[s.stage];
      t.innerHTML = `<b>${what}</b> <span>${s.stage === "restarting" ? "" : (s.pct || 0) + "%"}</span>`; bs.innerHTML = "";
      el.hidden = false; return;
    }
    if (s.stage === "done") {
      t.innerHTML = `<b>Updated to ${esc(s.current)}</b> <span>· close and reopen the app</span>`;
      bs.innerHTML = `<button data-a="x">OK</button>`; el.hidden = false; return;
    }
    if (s.stage === "error") {
      t.innerHTML = `<b>Update didn't finish</b> <span>· ${esc(s.error)} · nothing was changed</span>`;
      t.title = s.error;
      bs.innerHTML = `<button data-a="x">Later</button><button class="go" data-a="go">Try again</button>`;
      el.hidden = false; return;
    }
    if (!s.available || isLater()) { el.hidden = true; return; }
    const first = (s.notes || "").split("\n").map(x => x.replace(/^[-*•\s]+/, "")).find(x => x.trim()) || "";
    t.innerHTML = `<b>Update ${esc(s.latest)}</b> <span>${first ? "· " + esc(first) : ""}</span>`; t.title = first;
    bs.innerHTML = (s.notes ? `<button data-a="notes">Notes</button>` : "") +
                   `<button data-a="x">Later</button><button class="go" data-a="go">Update</button>`;
    $(".notes").textContent = s.notes || "";
    el.hidden = false;
  }
  function poll() {
    if (polling) return;
    polling = setInterval(() => fetch("/api/update").then(r => r.json()).then(s => {
      show(s);
      if (!["downloading", "installing", "packages", "restarting"].includes(s.stage)) { clearInterval(polling); polling = null; }
    }).catch(() => {}), 800);   // during "restarting" the server goes away; the window closes on its own
  }
  el.addEventListener("click", e => {
    const a = e.target.closest("button")?.dataset.a; if (!a) return;
    if (a === "x") later();
    if (a === "notes") $(".notes").hidden = !$(".notes").hidden;
    if (a === "go") post("/api/update/apply").then(s => {
      if (!s.ok && s.error) { $(".t").innerHTML = `<b>Not now</b> <span>· ${esc(s.error)}</span>`; return; }
      show(s); poll();
    }).catch(() => {});
  });
  // the server checks on start; ask again shortly after in case that check was still in flight
  const load = () => fetch("/api/update").then(r => r.json()).then(s => { if (s.enabled) show(s); return s; }).catch(() => ({}));
  load().then(s => { if (s.enabled && !s.available) setTimeout(load, 5000); });
})();
