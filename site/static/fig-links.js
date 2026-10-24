/* fig-links.js — evidence-linking invariant (SITE.md item 1, page half).
   render.py inlines figure SVGs (question qN, metrics m-*, experiment exp-*);
   figures.py tags marks with gid "ep:<episode id>[:t:<event index>]". This makes
   every tagged mark inside a <figure> a deep link into the transcript reader:
   episodes.html?e=<id>[&t=<i>]. Tile-grid thumbnails inline the same SVGs
   OUTSIDE a <figure>, so their marks are deliberately not wired — the whole
   tile is one anchor to its detail page.
   Runs on index.html (root) and on the fig/ and experiments/ detail pages, so
   the reader URL is prefixed with <body data-root> ("" at root, "../" below).
   Enhancement only: without JS (and in print) the marks render unchanged. */
(function () {
  "use strict";

  var GID = /^ep:(.+?)(?::t:(\d+))?$/;

  function readerUrl(root, gid) {
    var m = GID.exec(gid);
    if (!m) return null;
    var url = root + "episodes.html?e=" + encodeURIComponent(m[1]);
    if (m[2] !== undefined) url += "&t=" + m[2];
    return url;
  }

  function onClick(url) {
    return function (event) {
      /* a mark click goes to its episode, not to the surrounding
         <a class="fig-more"> whole-figure detail link */
      event.preventDefault();
      event.stopPropagation();
      window.location = url;
    };
  }

  function init() {
    var root = (document.body && document.body.getAttribute("data-root")) || "";
    var marks = document.querySelectorAll('figure svg [id^="ep:"]');
    for (var i = 0; i < marks.length; i++) {
      var url = readerUrl(root, marks[i].getAttribute("id"));
      if (!url) continue;
      marks[i].classList.add("fig-link"); /* cursor + hover stroke, site.css */
      marks[i].addEventListener("click", onClick(url));
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init(); /* script is deferred, so the inlined SVGs are already parsed */
  }
})();
