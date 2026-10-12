export function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

export const ICON = {
  incident: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.2"/><path d="M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4"/></svg>',
  map: '<svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 12.5l9 5 9-5"/><path d="M3 16.5l9 5 9-5"/></svg>',
  access: '<svg viewBox="0 0 24 24"><circle cx="5" cy="18" r="2.4"/><circle cx="19" cy="6" r="2.4"/><path d="M7 16.5c3-1.5 2-6 5-7.5s5 0 5-1"/></svg>',
  observability: '<svg viewBox="0 0 24 24"><path d="M3 20l6-9 4 5 3-4 5 8z"/><path d="M4 4l5 5M7 3l4 4" /></svg>',
  evidence: '<svg viewBox="0 0 24 24"><circle cx="6" cy="5" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="6" cy="19" r="2"/><path d="M6 7v3M6 14v3M10 5h10M10 12h10M10 19h7"/></svg>',
  operations: '<svg viewBox="0 0 24 24"><path d="M5 21V4M5 4h11l-2 4 2 4H5"/></svg>',
  methodology: '<svg viewBox="0 0 24 24"><path d="M4 5h6a2 2 0 012 2v13a2 2 0 00-2-2H4zM20 5h-6a2 2 0 00-2 2v13a2 2 0 012-2h6z"/></svg>',
  blocked: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6"/><path d="M3.8 12.2l8.4-8.4"/></svg>',
  open: '<svg viewBox="0 0 16 16"><path d="M2 12c3-1 4-8 7-8s3 5 5 5"/></svg>',
  unobservable: '<svg viewBox="0 0 16 16"><path d="M2 12l4-7 3 4 2-2 3 5z"/><path d="M1 1l14 14" /></svg>',
  settlements: '<svg viewBox="0 0 16 16"><path d="M2 14V8l4-3 4 3v6zM10 14V6l2-2 2 2v8"/></svg>',
  close: '<svg viewBox="0 0 16 16"><path d="M3 3l10 10M13 3L3 13"/></svg>',
  layers: '<svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
  eyeoff: '<svg viewBox="0 0 24 24"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><path d="M3 3l18 18"/></svg>',
  home: '<svg viewBox="0 0 24 24"><path d="M4 11l8-7 8 7v9H4z"/></svg>',
};

export function badge(cls, text) {
  const inner = cls === "unobservable" || cls === "POSSIBLE_CUTOFF" ? `<span>${text}</span>` : text;
  return `<span class="badge ${cls}">${inner}</span>`;
}
