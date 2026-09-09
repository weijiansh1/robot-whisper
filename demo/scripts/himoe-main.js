/* ===================================================================
   HiMoE story page — interactions
   nav state · scroll-reveal · theme · paired videos · cite copy
   (same engine as the RAD/ARM project pages)
   =================================================================== */
(function () {
  'use strict';

  var root = document.documentElement;
  var $  = function (s, c) { return (c || document).querySelector(s); };
  var $$ = function (s, c) { return Array.prototype.slice.call((c || document).querySelectorAll(s)); };

  /* ---------------- nav: stuck state ---------------- */
  var nav = $('#nav');
  var onScroll = function () { nav.classList.toggle('is-stuck', window.scrollY > 12); };
  onScroll(); window.addEventListener('scroll', onScroll, { passive: true });

  /* ---------------- mobile menu --------------------- */
  var navToggle = $('#nav-toggle'), navLinks = $('#nav-links');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    navLinks.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') {
        nav.classList.remove('is-open'); navToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ---------------- scroll reveal ------------------- */
  var reveals = $$('.reveal');
  if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    reveals.forEach(function (el) {                 // index within its reveal-group, for an ordered cascade
      var sibs = Array.prototype.filter.call(el.parentNode.children, function (c) { return c.classList.contains('reveal'); });
      el._ri = sibs.indexOf(el);
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.style.transitionDelay = Math.min(e.target._ri * 70, 360) + 'ms';
          e.target.classList.add('in'); io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add('in'); });
  }

  /* ---------------- theme toggle -------------------- */
  var meta = $('meta[name="theme-color"]');
  function applyTheme(t) {
    root.dataset.theme = t;
    if (meta) meta.setAttribute('content', t === 'light' ? '#F1EBDD' : '#0A0E15');
    try { localStorage.setItem('himoe-theme', t); } catch (e) {}
    if (window.radHero) window.radHero.setTheme(t);
    if (window.radFilm) window.radFilm.setTheme(t);
    if (window.radRescue) window.radRescue.setTheme(t);
    if (window.radPhase) window.radPhase.setTheme(t);
    if (window.radDuo) window.radDuo.setTheme(t);
  }
  var savedTheme; try { savedTheme = localStorage.getItem('himoe-theme'); } catch (e) {}
  if (savedTheme) applyTheme(savedTheme);
  $('#theme-toggle').addEventListener('click', function () {
    applyTheme(root.dataset.theme === 'light' ? 'dark' : 'light');
  });

  /* ---------------- paired videos: autoplay-loop on scroll, shared scrubber ---- */
  var vidReduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  $$('.vidwrap').forEach(function (wr) {
    var vids = $$('video', wr), btn = $('.vidplay', wr);
    var bar = $('.vidbar', wr), fill = bar && bar.querySelector('i');
    if (!vids.length) return;
    var master = vids[0], playing = false, scrubbing = false, wasPlaying = false;
    function pickMaster() {                              // the longer clip drives the shared timeline
      master = vids.reduce(function (a, b) { return (b.duration || 0) > (a.duration || 0) ? b : a; });
    }
    vids.forEach(function (v) { v.muted = true; v.addEventListener('loadedmetadata', pickMaster); });
    function playAll() {
      vids.forEach(function (v) {
        if (v !== master && v.duration && v.currentTime >= v.duration - 0.05) return;   // holds its final frame
        v.play().catch(function () {});
      });
      playing = true;
      if (btn) { btn.textContent = '❚❚'; btn.classList.add('is-dim'); }
    }
    function pauseAll() {
      vids.forEach(function (v) { v.pause(); });
      playing = false;
      if (btn) { btn.textContent = '▶'; btn.classList.remove('is-dim'); }
    }
    vids.forEach(function (v) {
      v.addEventListener('ended', function () {          // loop: when the master ends, both restart
        if (v === master && playing && !scrubbing) {
          vids.forEach(function (x) { x.currentTime = 0; });
          playAll();
        }
      });
    });
    (function tick() {                                    // progress fill
      if (fill && master && master.duration) fill.style.width = (master.currentTime / master.duration * 100) + '%';
      requestAnimationFrame(tick);
    })();
    function seekTo(clientX) {
      if (!bar || !master || !master.duration) return;
      var r = bar.getBoundingClientRect();
      var u = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      var t = u * master.duration;
      vids.forEach(function (v) { v.currentTime = Math.min(t, (v.duration || t) - 0.02); });
    }
    if (bar) {
      bar.addEventListener('pointerdown', function (e) {
        scrubbing = true; wasPlaying = playing; pauseAll();
        try { bar.setPointerCapture(e.pointerId); } catch (err) {}
        seekTo(e.clientX); e.preventDefault();
      });
      bar.addEventListener('pointermove', function (e) { if (scrubbing) seekTo(e.clientX); });
      bar.addEventListener('pointerup', function () { scrubbing = false; if (wasPlaying) playAll(); });
      bar.addEventListener('pointercancel', function () { scrubbing = false; });
    }
    function toggle() { playing ? pauseAll() : playAll(); }
    if (btn) btn.addEventListener('click', function (e) { e.stopPropagation(); toggle(); });
    wr.addEventListener('click', function (e) {
      if (e.target === btn || (bar && bar.contains(e.target))) return;
      toggle();
    });
    if (!vidReduce && 'IntersectionObserver' in window) {  // autoplay in view, pause out of view
      new IntersectionObserver(function (es) {
        es.forEach(function (en) {
          if (en.isIntersecting) { if (!playing && !scrubbing) playAll(); }
          else if (playing) pauseAll();
        });
      }, { threshold: 0.35 }).observe(wr);
    }
  });

  /* ---------------- cite: copy constants --------------- */
  var copyBtn = $('#bib-copy');
  if (copyBtn) copyBtn.addEventListener('click', function () {
    var txt = $('#bibtex').textContent;
    var done = function () {
      copyBtn.classList.add('is-done');
      var prev = copyBtn.textContent; copyBtn.textContent = '✓ Copied';
      setTimeout(function () { copyBtn.classList.remove('is-done'); copyBtn.textContent = prev; }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(done, function(){ fallback(txt); done(); });
    } else { fallback(txt); done(); }
  });
  function fallback(txt) {
    var ta = document.createElement('textarea'); ta.value = txt;
    ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta);
    ta.select(); try { document.execCommand('copy'); } catch (e) {} document.body.removeChild(ta);
  }
})();
