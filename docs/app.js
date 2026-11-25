"use strict";
// Static viewer for the central views exported by web_views.py / web_views_extra.py
// (data/manifest.json + data/sampleNN/*.png).
(function () {
  const PLANE_LABEL = { axial: "Axial", coronal: "Coronal", sagittal: "Sagittal" };
  const MODES = [
    { key: "gray", label: "Gray values" },
    { key: "err", label: "Error vs 60 min" },
    { key: "seg", label: "Segmentation" },
  ];
  const PIPES = [
    { key: "otsu2_raw", label: "Otsu, 2 classes" },
    { key: "otsu2_med3", label: "Median + Otsu, 2 classes" },
    { key: "otsu3_raw", label: "Multi-Otsu, 3 classes" },
    { key: "otsu3_med3", label: "Median + multi-Otsu, 3 classes" },
  ];
  // volumes available in each mode
  const MODE_VOLS = {
    gray: ["ref60", "fast", "int8", "redcnn", "finetuned", "scratch", "manual"],
    err: ["fast", "int8", "redcnn", "finetuned", "scratch"],
    seg: ["manual", "ref60", "fast", "int8", "redcnn", "finetuned", "scratch"],
  };
  const SEG_LABEL = { ref60: "60 min (automatic)", manual: "Manual segmentation (60 min)" };

  const state = {
    man: null, sample: 0, plane: "axial", mode: "gray", pipe: "otsu3_raw", layout: "volumes",
    on: new Set(), single: "scratch", lb: -1, segCmp: false,
  };
  const $ = (id) => document.getElementById(id);

  // Segmentation PNGs encode the comparison with the manual reference (black both, red added, blue missed,
  // light gray solid). Without "compare", they are recoloured to a plain mask: pore black, solid light gray.
  const plainCache = new Map();
  function plainSeg(url) {
    if (!plainCache.has(url)) {
      plainCache.set(url, new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => {
          const c = document.createElement("canvas");
          c.width = im.naturalWidth; c.height = im.naturalHeight;
          const ctx = c.getContext("2d");
          ctx.drawImage(im, 0, 0);
          const d = ctx.getImageData(0, 0, c.width, c.height), p = d.data;
          for (let i = 0; i < p.length; i += 4) {
            const r = p[i], g = p[i + 1], b = p[i + 2];
            if (r > 150 && g < 100 && b < 100) { p[i] = p[i + 1] = p[i + 2] = 0; }          // added -> pore
            else if (b > 150 && r < 100) { p[i] = p[i + 1] = p[i + 2] = 242; }               // missed -> solid
          }
          ctx.putImageData(d, 0, 0);
          resolve(c.toDataURL("image/png"));
        };
        im.onerror = reject;
        im.src = url;
      }));
    }
    return plainCache.get(url);
  }

  function setSrc(img, url) {
    if (state.mode === "seg" && !state.segCmp && url.includes("/seg_")) {
      img.removeAttribute("src");
      plainSeg(url).then((d) => { img.src = d; }).catch(() => { img.src = url; });
    } else {
      img.src = url;
    }
  }

  function store(key, val) {
    try {
      if (val === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, val);
    } catch (e) { return null; }
    return null;
  }

  function vlabel(key) {
    if (state.mode === "seg" && SEG_LABEL[key]) return SEG_LABEL[key];
    const v = state.man.volumes.find((x) => x.key === key);
    return v ? v.label : key;
  }

  function src(s, key, plane) {
    if (state.mode === "err") return `data/${s.dir}/err_${key}_${plane}.png`;
    if (state.mode === "seg" && key !== "manual") return `data/${s.dir}/seg_${state.pipe}_${key}_${plane}.png`;
    return `data/${s.dir}/${key}_${plane}.png`;
  }

  // horizontal extent of each plane in voxels (shape is z, x, y)
  function widthVox(s, plane) { return plane === "coronal" ? s.shape_zxy[1] : s.shape_zxy[2]; }

  function badge(s, key) {
    const phiMan = s.porosity_manual != null ? `φ = ${s.porosity_manual.toFixed(3)}` : "";
    if (key === "manual") return phiMan;
    if (state.mode === "seg") {
      const p = s.porosity && s.porosity[state.pipe] && s.porosity[state.pipe][key];
      const d = s.dice && s.dice[state.pipe] && s.dice[state.pipe][key];
      return [p != null ? `φ = ${p.toFixed(3)}` : "", d != null ? `Dice ${d.toFixed(3)}` : ""].filter(Boolean).join(" · ");
    }
    const r = s.metrics && s.metrics[key];
    return r ? `PSNR ${r.psnr.toFixed(2)} dB · SSIM ${r.ssim.toFixed(3)}` : "";
  }

  function chip(text, attr, onClick, data) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = text;
    b.setAttribute(attr, "false");
    if (data) Object.assign(b.dataset, data);
    b.addEventListener("click", onClick);
    return b;
  }

  function fillChips(id, items, attr, isOn, onClick) {
    const el = $(id); el.innerHTML = "";
    items.forEach((it) => {
      const b = chip(it.label, attr, () => onClick(it.key), { key: it.key });
      b.setAttribute(attr, String(isOn(it.key)));
      el.appendChild(b);
    });
  }

  function modeVols() { return MODE_VOLS[state.mode].filter((k) => state.man.volumes.some((v) => v.key === k)); }

  function shownVols() { return modeVols().filter((k) => state.on.has(k)); }

  function persist() {
    store("viewer.v2", JSON.stringify({
      sample: state.sample, plane: state.plane, mode: state.mode, pipe: state.pipe, layout: state.layout,
      on: [...state.on], single: state.single, segCmp: state.segCmp,
    }));
    const q = new URLSearchParams({
      sample: String(state.man.samples[state.sample].id), plane: state.plane, mode: state.mode,
    });
    if (state.mode === "seg") { q.set("pipeline", state.pipe); if (state.segCmp) q.set("compare", "1"); }
    if (state.layout === "samples") { q.set("layout", "samples"); q.set("volume", state.single); }
    try { history.replaceState(null, "", "?" + q.toString()); } catch (e) { /* file:// */ }
  }

  function renderControls() {
    const m = state.man;
    fillChips("samples", m.samples.map((s, i) => ({ key: String(i), label: String(s.id) })),
      "aria-selected", (k) => Number(k) === state.sample, (k) => { state.sample = Number(k); render(); });
    $("samplesGroup").hidden = state.layout === "samples";
    fillChips("layout", [{ key: "volumes", label: "One sample, several volumes" }, { key: "samples", label: "One volume, all samples" }],
      "aria-selected", (k) => k === state.layout, (k) => { state.layout = k; render(); });
    fillChips("modes", MODES, "aria-selected", (k) => k === state.mode, (k) => { state.mode = k; fixSingle(); render(); });
    fillChips("planes", m.planes.map((p) => ({ key: p, label: PLANE_LABEL[p] || p })), "aria-selected",
      (k) => k === state.plane, (k) => { state.plane = k; render(); });
    $("pipeGroup").hidden = state.mode !== "seg";
    fillChips("pipes", PIPES, "aria-selected", (k) => k === state.pipe, (k) => { state.pipe = k; render(); });
    fillChips("segview", [{ key: "plain", label: "Pore / solid" }, { key: "cmp", label: "Compare with manual" }],
      "aria-selected", (k) => (k === "cmp") === state.segCmp, (k) => { state.segCmp = k === "cmp"; render(); });
    const vols = modeVols().map((k) => ({ key: k, label: vlabel(k) }));
    if (state.layout === "volumes") {
      $("volLab").textContent = "Volumes";
      fillChips("volumes", vols, "aria-pressed", (k) => state.on.has(k), (k) => {
        if (state.on.has(k)) { if (shownVols().length > 1) state.on.delete(k); } else state.on.add(k);
        render();
      });
      const all = chip("All", "aria-pressed", () => { modeVols().forEach((k) => state.on.add(k)); render(); });
      all.classList.add("chip-ghost");
      $("volumes").appendChild(all);
    } else {
      $("volLab").textContent = "Volume";
      fillChips("volumes", vols, "aria-selected", (k) => k === state.single, (k) => { state.single = k; render(); });
    }
    // compare selects
    ["cmpA", "cmpB"].forEach((id, i) => {
      const sel = $(id), prev = sel.value;
      sel.innerHTML = "";
      vols.forEach((v) => {
        const o = document.createElement("option"); o.value = v.key; o.textContent = v.label; sel.appendChild(o);
      });
      const want = prev && vols.some((v) => v.key === prev) ? prev : (i === 0 ? vols[Math.min(1, vols.length - 1)].key : "scratch");
      sel.value = vols.some((v) => v.key === want) ? want : vols[vols.length - 1].key;
    });
  }

  function fixSingle() {
    if (!modeVols().includes(state.single)) state.single = modeVols().includes("scratch") ? "scratch" : modeVols()[0];
  }

  function legend() {
    const L = $("legend");
    if (state.mode === "err") {
      const lim = state.man.err_lim || 2500;
      L.innerHTML = `<span class="rampbox"><span class="ramp"></span><span class="ticks"><span>&minus;${lim}</span><span>0</span><span>+${lim}</span></span></span>` +
        `<span class="lt">Volume &minus; reference (60 min), gray levels</span>`;
    } else if (state.mode === "seg") {
      const items = state.segCmp
        ? [["#000000", "Pore in both"], ["#d62728", "Pore only in the automatic mask (added)"],
          ["#1f77b4", "Pore only in the manual reference (missed)"], ["#f2f2f2", "Solid in both"]]
        : [["#000000", "Pore"], ["#f2f2f2", "Solid"]];
      L.innerHTML = items.map(([c, t]) => `<span class="sw"><i style="background:${c}"></i>${t}</span>`).join("");
    } else {
      L.innerHTML = "";
    }
    L.hidden = state.mode === "gray";
  }

  function card(s, key, title) {
    const c = document.createElement("article"); c.className = "card";
    const h = document.createElement("header");
    h.innerHTML = `<h3></h3><span class="m"></span>`;
    h.querySelector("h3").textContent = title;
    h.querySelector(".m").textContent = badge(s, key);
    const f = document.createElement("div"); f.className = "frame";
    const img = document.createElement("img");
    img.loading = "lazy"; img.decoding = "async";
    setSrc(img, src(s, key, state.plane));
    img.alt = `${s.label}, ${vlabel(key)}, ${PLANE_LABEL[state.plane]} view`;
    const px = s.image_px && s.image_px[state.plane];
    if (px) { img.width = px[0]; img.height = px[1]; }
    img.addEventListener("error", () => { f.classList.add("missing"); });
    f.appendChild(img);
    const frac = (5000 / s.voxel_um) / widthVox(s, state.plane);
    const sb = document.createElement("div"); sb.className = "scalebar";
    sb.style.width = (100 * frac).toFixed(2) + "%"; sb.innerHTML = "<span>5 mm</span>";
    f.appendChild(sb);
    f.addEventListener("click", () => openLb(s, key));
    c.appendChild(h); c.appendChild(f);
    return c;
  }

  function render() {
    const m = state.man, s = m.samples[state.sample];
    renderControls(); legend();
    const g = $("grid"); g.innerHTML = "";
    g.classList.toggle("tall", state.plane !== "axial");
    if (state.layout === "volumes") {
      const [z, x, y] = s.shape_zxy;
      $("info").textContent = `${s.label} · ${PLANE_LABEL[state.plane]} central slice · crop ${x} × ${y} × ${z} voxels · ` +
        `voxel size ${s.voxel_um.toFixed(1)} µm · gray window ${s.window[0]}–${s.window[1]}` +
        (s.porosity_manual != null ? ` · manual porosity φ = ${s.porosity_manual.toFixed(3)}` : "") +
        (state.mode === "seg" ? ` · ${PIPES.find((p) => p.key === state.pipe).label}; porosity φ and Dice computed in 3D over the whole crop` : "");
      shownVols().forEach((k) => g.appendChild(card(s, k, vlabel(k))));
    } else {
      $("info").textContent = `${vlabel(state.single)} · ${PLANE_LABEL[state.plane]} central slice of every sample` +
        (state.mode === "seg" ? ` · ${PIPES.find((p) => p.key === state.pipe).label}` : "") +
        " · each sample has its own gray window";
      m.samples.forEach((smp) => g.appendChild(card(smp, state.single, smp.label)));
    }
    $("compare").hidden = state.layout !== "volumes";
    renderSwipe();
    persist();
    if (state.lb >= 0 && !$("lightbox").hidden) showLb();
  }

  function renderSwipe() {
    const s = state.man.samples[state.sample];
    const a = $("cmpA").value, b = $("cmpB").value;
    if (!a || !b) return;
    setSrc($("swipeA"), src(s, a, state.plane)); $("swipeA").alt = vlabel(a);
    setSrc($("swipeB"), src(s, b, state.plane)); $("swipeB").alt = vlabel(b);
    $("swipeLabA").textContent = vlabel(a); $("swipeLabB").textContent = vlabel(b);
    $("swipe").classList.toggle("tall", state.plane !== "axial");
    setSwipe($("swipeRange").value);
  }

  function setSwipe(p) {
    $("swipeTop").style.clipPath = `inset(0 ${100 - p}% 0 0)`;
    $("swipeBar").style.left = p + "%";
  }

  // lightbox: steps through the cards currently shown
  let lbItems = [];
  function openLb(s, key) {
    lbItems = state.layout === "volumes" ? shownVols().map((k) => [state.sample, k]) :
      state.man.samples.map((_, i) => [i, state.single]);
    state.lb = lbItems.findIndex(([i, k]) => state.man.samples[i] === s && k === key);
    $("lightbox").hidden = false; showLb();
  }
  function showLb() {
    if (!lbItems.length) return closeLb();
    state.lb = (state.lb + lbItems.length) % lbItems.length;
    const [i, k] = lbItems[state.lb], s = state.man.samples[i];
    setSrc($("lbImg"), src(s, k, state.plane));
    $("lbImg").alt = `${s.label}, ${vlabel(k)}`;
    const b = badge(s, k);
    $("lbCap").textContent = `${s.label} · ${PLANE_LABEL[state.plane]} · ${vlabel(k)}` + (b ? ` · ${b}` : "");
  }
  function closeLb() { state.lb = -1; $("lightbox").hidden = true; }

  function keys(e) {
    const tag = e.target && e.target.tagName;
    if (tag === "SELECT" || tag === "INPUT") return;
    const lb = !$("lightbox").hidden, n = state.man.samples.length;
    if (e.key === "Escape" && lb) closeLb();
    else if (e.key === "ArrowRight") { if (lb) { state.lb++; showLb(); } else if (state.layout === "volumes") { state.sample = (state.sample + 1) % n; render(); } }
    else if (e.key === "ArrowLeft") { if (lb) { state.lb--; showLb(); } else if (state.layout === "volumes") { state.sample = (state.sample - 1 + n) % n; render(); } }
  }

  function init(man) {
    state.man = man;
    let saved = null;
    try { saved = JSON.parse(store("viewer.v2") || "null"); } catch (e) { saved = null; }
    if (saved) {
      if (saved.sample < man.samples.length) state.sample = saved.sample;
      if (man.planes.includes(saved.plane)) state.plane = saved.plane;
      if (MODE_VOLS[saved.mode]) state.mode = saved.mode;
      if (PIPES.some((p) => p.key === saved.pipe)) state.pipe = saved.pipe;
      if (saved.layout === "samples" || saved.layout === "volumes") state.layout = saved.layout;
      if (Array.isArray(saved.on)) state.on = new Set(saved.on);
      if (saved.single) state.single = saved.single;
      state.segCmp = saved.segCmp === true;
    }
    const q = new URLSearchParams(location.search);
    const qs = parseInt(q.get("sample"), 10);
    const qi = man.samples.findIndex((s) => s.id === qs);
    if (qi >= 0) state.sample = qi;
    if (man.planes.includes(q.get("plane"))) state.plane = q.get("plane");
    if (MODE_VOLS[q.get("mode")]) state.mode = q.get("mode");
    if (PIPES.some((p) => p.key === q.get("pipeline"))) state.pipe = q.get("pipeline");
    if (q.get("layout") === "samples") state.layout = "samples";
    if (q.get("volume")) state.single = q.get("volume");
    if (q.has("compare")) state.segCmp = q.get("compare") === "1";
    if (!man.has_extra) state.mode = "gray";
    if (![...state.on].some((k) => man.volumes.some((v) => v.key === k))) state.on = new Set(man.volumes.map((v) => v.key));
    fixSingle();
    if (!man.has_extra) $("modesGroup").hidden = true;
    $("cmpA").addEventListener("change", renderSwipe);
    $("cmpB").addEventListener("change", renderSwipe);
    $("swipeRange").addEventListener("input", (e) => setSwipe(e.target.value));
    $("lbClose").addEventListener("click", closeLb);
    $("lbPrev").addEventListener("click", () => { state.lb--; showLb(); });
    $("lbNext").addEventListener("click", () => { state.lb++; showLb(); });
    $("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") closeLb(); });
    document.addEventListener("keydown", keys);
    render();
  }

  fetch("data/manifest.json", { cache: "no-cache" })
    .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(init)
    .catch((e) => { $("info").textContent = "Could not load data/manifest.json (" + e.message + ")."; });
})();
