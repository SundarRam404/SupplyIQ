export default function HeroIllustration() {
  return (
    <svg viewBox="0 0 600 420" className="hero-illustration" role="img" aria-label="Supply chain intelligence dashboard illustration">
      <defs>
        <linearGradient id="panel" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#11203e" />
          <stop offset="100%" stopColor="#0a1224" />
        </linearGradient>
        <linearGradient id="accent" x1="0%" x2="100%">
          <stop offset="0%" stopColor="#7c3aed" />
          <stop offset="50%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#22c55e" />
        </linearGradient>
      </defs>
      <rect x="40" y="40" width="520" height="320" rx="28" fill="url(#panel)" stroke="rgba(255,255,255,0.16)" />
      <rect x="72" y="78" width="180" height="96" rx="18" fill="#121d36" stroke="rgba(255,255,255,0.08)" />
      <rect x="272" y="78" width="250" height="96" rx="18" fill="#121d36" stroke="rgba(255,255,255,0.08)" />
      <rect x="72" y="194" width="450" height="126" rx="18" fill="#0d172d" stroke="rgba(255,255,255,0.08)" />
      <path d="M105 280 C155 240, 185 260, 230 215 S330 205, 380 170 S455 175, 500 130" fill="none" stroke="url(#accent)" strokeWidth="8" strokeLinecap="round" />
      <circle cx="230" cy="215" r="8" fill="#8b5cf6" />
      <circle cx="380" cy="170" r="8" fill="#06b6d4" />
      <circle cx="500" cy="130" r="8" fill="#22c55e" />
      <rect x="95" y="100" width="75" height="12" rx="6" fill="#7c3aed" opacity="0.9" />
      <rect x="95" y="124" width="110" height="10" rx="5" fill="#314266" />
      <rect x="95" y="145" width="84" height="10" rx="5" fill="#314266" />
      <rect x="292" y="101" width="90" height="12" rx="6" fill="#06b6d4" opacity="0.9" />
      <rect x="292" y="126" width="190" height="10" rx="5" fill="#314266" />
      <rect x="292" y="148" width="140" height="10" rx="5" fill="#314266" />
      <rect x="438" y="228" width="56" height="56" rx="14" fill="#132342" />
      <path d="M455 255 L468 268 L486 242" fill="none" stroke="#22c55e" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
      <text x="96" y="240" fill="#cbd5e1" fontSize="18" fontFamily="Arial">Demand trajectory</text>
      <text x="96" y="264" fill="#64748b" fontSize="12" fontFamily="Arial">Forecasting • risk detection • contracts • decisions</text>
    </svg>
  )
}
