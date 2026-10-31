/**
 * config/images.js - Centralized Asset & Visual Texture Registry
 * Warm orange glass meditation aesthetic matching reference screenshot.
 * Pure vector SVG & CSS data URLs for 100% offline reliability.
 * Can be swapped with custom JPG/PNG image URLs anytime.
 */

// 1. Rust Canyon Background for Home Screen (Phone 1)
// Layered warm rust, terracotta, and deep ember sandstone canyon landscape
export const homeCanyonSvg = `data:image/svg+xml;utf8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 390 844" width="100%" height="100%" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4A150A"/>
      <stop offset="35%" stop-color="#7A2412"/>
      <stop offset="65%" stop-color="#C2410C"/>
      <stop offset="100%" stop-color="#FF5A1F"/>
    </linearGradient>
    <radialGradient id="sunGlow" cx="50%" cy="40%" r="55%">
      <stop offset="0%" stop-color="#FFA07A" stop-opacity="0.85"/>
      <stop offset="45%" stop-color="#FF5A1F" stop-opacity="0.45"/>
      <stop offset="85%" stop-color="#7A2412" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="canyonLeft" x1="0" y1="0" x2="1" y2="0.8">
      <stop offset="0%" stop-color="#2D0B05"/>
      <stop offset="50%" stop-color="#5C1B0D"/>
      <stop offset="100%" stop-color="#8E2D16"/>
    </linearGradient>
    <linearGradient id="canyonRight" x1="1" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3B0E07"/>
      <stop offset="40%" stop-color="#702210"/>
      <stop offset="85%" stop-color="#A33718"/>
      <stop offset="100%" stop-color="#E0531E"/>
    </linearGradient>
    <linearGradient id="bottomShadow" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="transparent"/>
      <stop offset="50%" stop-color="rgba(15, 6, 4, 0.45)"/>
      <stop offset="100%" stop-color="rgba(12, 4, 2, 0.94)"/>
    </linearGradient>
  </defs>

  <!-- Canyon Sky & Warm Atmosphere -->
  <rect width="390" height="844" fill="url(#skyGrad)"/>
  <circle cx="210" cy="340" r="280" fill="url(#sunGlow)"/>

  <!-- Distant Dunes / Rock Formations -->
  <path d="M0,420 Q120,380 220,410 T390,390 L390,844 L0,844 Z" fill="#6E1F0E" opacity="0.65"/>
  <path d="M0,480 Q160,430 280,470 T390,450 L390,844 L0,844 Z" fill="#52160A" opacity="0.85"/>

  <!-- Dramatic Left Canyon Arch Wall -->
  <path d="M-20,-10 C90,80 140,240 110,380 C80,500 20,620 -40,700 L-40,-10 Z" fill="url(#canyonLeft)"/>
  <path d="M-20,-10 C60,110 90,260 70,390 C50,510 0,620 -50,680 L-50,-10 Z" fill="#1F0703" opacity="0.75"/>

  <!-- Dramatic Right Canyon Sunlit Cliff -->
  <path d="M410,-10 C320,120 250,280 270,440 C290,580 370,710 430,780 L430,-10 Z" fill="url(#canyonRight)"/>
  <path d="M410,-10 C340,140 290,300 310,430 C330,550 390,670 440,750 L440,-10 Z" fill="#8C2C14" opacity="0.5"/>

  <!-- Soft Fog / Glow at Canyon Floor -->
  <ellipse cx="200" cy="520" rx="160" ry="90" fill="#FF773D" opacity="0.25"/>

  <!-- Dark Bottom Vignette for Crisp Text Legibility -->
  <rect width="390" height="844" fill="url(#bottomShadow)"/>
</svg>
`)}`;

// 2. Resting Portrait Background for Recording Screen (Phone 3)
// Serene warm orange/amber graded portrait silhouette of a person resting
export const restingPortraitSvg = `data:image/svg+xml;utf8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 390 844" width="100%" height="100%" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4F170B"/>
      <stop offset="35%" stop-color="#7B2513"/>
      <stop offset="70%" stop-color="#4A1409"/>
      <stop offset="100%" stop-color="#1F0703"/>
    </linearGradient>
    <radialGradient id="portraitLight" cx="65%" cy="32%" r="50%">
      <stop offset="0%" stop-color="#FFA270" stop-opacity="0.95"/>
      <stop offset="30%" stop-color="#FF5A1F" stop-opacity="0.6"/>
      <stop offset="75%" stop-color="#7B2513" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="skinGlow" x1="0" y1="0" x2="1" y2="0.6">
      <stop offset="0%" stop-color="#FFE0D2"/>
      <stop offset="40%" stop-color="#FFA579"/>
      <stop offset="100%" stop-color="#8C2B14"/>
    </linearGradient>
    <linearGradient id="vignette" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgba(15, 5, 3, 0.4)"/>
      <stop offset="45%" stop-color="transparent"/>
      <stop offset="80%" stop-color="rgba(12, 4, 2, 0.88)"/>
      <stop offset="100%" stop-color="rgba(8, 2, 1, 0.96)"/>
    </linearGradient>
  </defs>

  <!-- Warm Graded Background -->
  <rect width="390" height="844" fill="url(#bgGrad)"/>
  <circle cx="250" cy="280" r="240" fill="url(#portraitLight)"/>

  <!-- Artistic Stylized Resting Profile Silhouette -->
  <!-- Hair & Head Contour -->
  <path d="M210,130 C270,125 320,165 330,225 C335,260 320,300 300,335 C285,360 270,390 280,430 C290,470 330,520 370,570 L390,844 L90,844 C120,740 140,650 150,580 C155,540 145,500 135,460 C125,410 135,360 150,330 C158,310 158,260 162,230 C170,160 190,135 210,130 Z" fill="#240A04" opacity="0.85"/>
  
  <!-- Face & Jawline with Warm Orange Rim Light -->
  <path d="M225,185 C245,190 265,205 272,225 C275,235 268,245 272,255 C276,262 284,268 283,276 C281,283 273,288 273,296 C272,306 280,314 278,324 C272,342 250,360 232,370 C220,378 215,400 216,420 C218,460 230,500 245,540 L160,560 C150,510 148,460 152,410 C155,370 170,340 185,320 C195,305 195,290 192,275 C188,255 190,230 198,210 C206,195 215,188 225,185 Z" fill="url(#skinGlow)" opacity="0.92"/>

  <!-- Closed Eye and Gentle Expression -->
  <path d="M252,248 Q262,254 270,250" stroke="#5A1A0C" stroke-width="2.5" stroke-linecap="round" fill="none"/>
  <!-- Soft Lip Contour -->
  <path d="M266,298 Q272,302 277,299" stroke="#681F0E" stroke-width="2" stroke-linecap="round" fill="none"/>

  <!-- Atmospheric Glow Overlays -->
  <ellipse cx="270" cy="270" rx="90" ry="120" fill="#FFA570" opacity="0.32"/>

  <!-- Dark Top and Bottom Vignette -->
  <rect width="390" height="844" fill="url(#vignette)"/>
</svg>
`)}`;

// 3. Card Artwork & Gradients for Record Screen (Phone 2)
export const cardAssets = {
  morning: {
    title: 'Morning Check',
    duration: '1 min',
    tag: 'Resting',
    bg: 'linear-gradient(135deg, #FF6B35 0%, #EA580C 50%, #C2410C 100%)',
    iconColor: '#FF5A1F'
  },
  workout: {
    title: 'After Workout',
    duration: '1 min',
    tag: 'After activity',
    bg: 'linear-gradient(135deg, #FDBA74 0%, #FB923C 50%, #EA580C 100%)',
    iconColor: '#EA580C'
  },
  calm: {
    title: 'Calm Down',
    duration: '2 min',
    tag: 'Guided breathing',
    bg: 'linear-gradient(135deg, #94A3B8 0%, #64748B 50%, #475569 100%)',
    iconColor: '#64748B'
  },
  night: {
    title: 'Night Check',
    duration: '1 min',
    tag: 'Resting',
    bg: 'linear-gradient(135deg, #312E81 0%, #4338CA 50%, #3730A3 100%)',
    iconColor: '#6366F1'
  }
};

// 4. Phone Placement Sternum Diagram (Inline SVG)
export const sternumDiagramSvg = `data:image/svg+xml;utf8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 180" width="100%" height="100%">
  <defs>
    <linearGradient id="bodySkin" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FED7AA"/>
      <stop offset="100%" stop-color="#FDBA74"/>
    </linearGradient>
    <linearGradient id="phoneGlow" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FF5A1F"/>
      <stop offset="100%" stop-color="#C2410C"/>
    </linearGradient>
  </defs>

  <!-- Torso & Sternum Outline -->
  <path d="M40,170 C45,110 65,70 95,50 C105,44 115,40 120,40 C125,40 135,44 145,50 C175,70 195,110 200,170 Z" fill="url(#bodySkin)" opacity="0.75"/>
  
  <!-- Collarbones -->
  <path d="M70,68 Q120,82 120,82 Q120,82 170,68" stroke="#EA580C" stroke-width="2.5" stroke-linecap="round" fill="none" opacity="0.6"/>

  <!-- Sternum (Breastbone) Centerline Guide -->
  <line x1="120" y1="85" x2="120" y2="145" stroke="#EA580C" stroke-width="2" stroke-dasharray="3,3" opacity="0.6"/>

  <!-- Flat Phone on Lower Sternum -->
  <g transform="translate(95, 96)">
    <!-- Pulse Ring Indicator -->
    <rect x="-6" y="-6" width="62" height="52" rx="12" fill="none" stroke="#FF5A1F" stroke-width="2" opacity="0.7">
      <animate attributeName="opacity" values="0.8;0.2;0.8" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="transform" type="scale" values="1;1.08;1" dur="2s" repeatCount="indefinite" transform-origin="25 20"/>
    </rect>
    <!-- Smartphone Outline -->
    <rect x="0" y="0" width="50" height="40" rx="8" fill="#1F1A17" stroke="#FFFFFF" stroke-width="1.8"/>
    <!-- Screen Accent -->
    <rect x="4" y="4" width="42" height="32" rx="5" fill="url(#phoneGlow)"/>
    <!-- Heart Icon on Phone Screen -->
    <path d="M25,23 L21.5,19.5 C20,18 20,15.5 21.5,14 C23,12.5 25.5,12.5 27,14 C28.5,12.5 31,12.5 32.5,14 C34,15.5 34,18 32.5,19.5 Z" fill="#FFFFFF" transform="translate(-2, -3) scale(0.9)"/>
  </g>
</svg>
`)}`;

export const IMAGES = {
  homeCanyon: homeCanyonSvg,
  restingPortrait: restingPortraitSvg,
  cards: cardAssets,
  sternumGuide: sternumDiagramSvg
};

export default IMAGES;

