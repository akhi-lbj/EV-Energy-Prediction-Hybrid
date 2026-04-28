---
name: Electric Grid Dark
colors:
  background: "#0f172a"
  surface: "#1e293b"
  surface-highlight: "#334155"
  primary: "#10b981"
  primary-hover: "#059669"
  on-primary: "#ffffff"
  secondary: "#3b82f6"
  on-secondary: "#ffffff"
  accent: "#f59e0b"
  text-main: "#f8fafc"
  text-muted: "#94a3b8"
  border: "#475569"
  error: "#ef4444"
typography:
  display:
    fontFamily: "Inter, sans-serif"
    fontSize: "48px"
    fontWeight: "700"
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Inter, sans-serif"
    fontSize: "24px"
    fontWeight: "600"
  body:
    fontFamily: "Inter, sans-serif"
    fontSize: "16px"
    fontWeight: "400"
  data-readout:
    fontFamily: "'JetBrains Mono', monospace"
    fontSize: "36px"
    fontWeight: "600"
  label:
    fontFamily: "Inter, sans-serif"
    fontSize: "12px"
    fontWeight: "600"
    textTransform: "uppercase"
    letterSpacing: "0.05em"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  section: "48px"
rounded:
  sm: "4px"
  md: "8px"
  lg: "12px"
  pill: "9999px"
shadows:
  glow-primary: "0 0 20px rgba(16, 185, 129, 0.25)"
  glow-secondary: "0 0 20px rgba(59, 130, 246, 0.25)"
  card: "0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.25)"
components:
  card:
    backgroundColor: "{colors.surface}"
    borderRadius: "{rounded.lg}"
    padding: "{spacing.lg}"
    boxShadow: "{shadows.card}"
    border: "1px solid {colors.border}"
  button-primary:
    backgroundColor: "{colors.primary}"
    color: "{colors.on-primary}"
    borderRadius: "{rounded.md}"
    padding: "{spacing.sm} {spacing.lg}"
    fontFamily: "{typography.label.fontFamily}"
    fontWeight: "600"
    boxShadow: "{shadows.glow-primary}"
  button-randomizer:
    backgroundColor: "transparent"
    color: "{colors.secondary}"
    borderRadius: "{rounded.pill}"
    padding: "{spacing.sm} {spacing.md}"
    border: "1px dashed {colors.secondary}"
    boxShadow: "{shadows.glow-secondary}"
  data-panel:
    backgroundColor: "#0b0f19"
    borderRadius: "{rounded.md}"
    padding: "{spacing.md}"
    borderLeft: "4px solid {colors.primary}"
---

## Brand & Style
This design system centers on an **Electric Grid Dark** aesthetic. It is tailored for high-tech data science and infrastructure dashboards, specifically focusing on probabilistic forecasting and real-time optimization. The visual language is highly technical, deeply analytical, and emphasizes absolute clarity of data over decorative elements.

The UI relies on a dark, high-contrast palette. Deep slate backgrounds simulate a command-center environment, while vibrant "electric" neon accents (greens, blues, and oranges) guide the user's eye to critical predictions, safety bounds, and interactive controls. 

## Colors
The color strategy uses deep, low-luminosity backgrounds to reduce eye strain, allowing vibrant data visualizations to pop.

- **Canvas & Surfaces:** The primary background is a deep slate (`#0f172a`), with slightly lighter elevated surfaces (`#1e293b`) for cards and panels.
- **Electric Accents:** 
  - **Primary (Electric Green):** Used for successful predictions, optimal states, and primary actions (e.g., "Run Forecast").
  - **Secondary (Neon Blue):** Used for analytical data, confidence intervals, and exploratory actions (e.g., "Auto-Fill Random Session").
  - **Accent (Urgency Orange):** Used for highlighting highly urgent sessions or constrained flexibilities.
- **Text:** Crisp white for primary headings and values, with a muted slate-gray for secondary labels and metadata to maintain strict visual hierarchy.

## Typography
The typographic system pairs a highly readable sans-serif with a technical monospace font for data readouts.

- **Primary Interface (Inter):** Used for all standard UI elements, labels, and body copy. It is clean, geometric, and modern.
- **Data Readouts (JetBrains Mono):** A strict monospace font is utilized specifically for displaying the ML outputs (e.g., "34.5 kWh", "[28.1 - 39.2 kWh]"). This ensures numbers align perfectly and evokes a terminal/code-driven aesthetic.
- **Labeling:** Form inputs and data metrics use the `label` token—small, uppercase, heavily tracked text to clearly denote *what* the user is looking at without competing with the data itself.

## Layout & Spacing
The layout is modular, relying on distinct `card` components to group related information.

- **Input vs. Output:** The UI is typically split into two distinct visual zones. The left/top zone houses the input features (Requested Energy, Connection Time, Available Minutes). The right/bottom zone acts as the analytical readout panel (Point Prediction, Lower Bound, Upper Bound, Urgency).
- **Density:** The dashboard uses generous padding (`24px` on cards) to prevent the highly technical data from feeling overwhelming or cluttered.

## Elevation & Depth
Depth is created through a combination of subtle drop shadows and neon "glow" effects.

- **Card Elevation:** Standard containers sit above the background using a soft, dark shadow to create physical separation.
- **Neon Glows:** Instead of traditional shadows, interactive elements (like the Primary Button or the Randomizer) emit a colored glow (`shadows.glow-primary`). This reinforces the "electric/energy" theme of the application.
- **Data Panels:** Specific numerical outputs are housed in recessed `data-panel` components. These use an ultra-dark background (`#0b0f19`) to mimic an LED screen readout, accented by a solid colored left-border indicating status.

## Core UI Elements

### The Prediction Readout Panel
The focal point of the UI. This area utilizes the `data-panel` component. It displays the **Expected Delivery (kWh)** in large, glowing `data-readout` typography. Sub-metrics, such as the 5th and 95th percentile confidence bounds, are displayed directly below in smaller, muted typography, providing a clear "Safety Net" visualization.

### Input Form
A clean, structured set of input fields allowing the user to tweak the EV session parameters manually. Inputs are styled with slate borders and subtle hover states.

### The Auto-Fill Randomizer Button
To improve user experience, the UI includes a "Fetch Random Session" button. Styled using the `button-randomizer` component (dashed border, neon blue text, pill shape), this button allows users to instantly populate the input form with realistic historical data without having to type out complex parameters manually.

### Execution Controls
The main action to trigger the probabilistic forecast is housed in a prominent `button-primary`. It utilizes the electric green color and glow, making it the obvious primary call-to-action on the screen.
