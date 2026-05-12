import sys

with open("main.py", "r", encoding="utf-8") as f:
    text = f.read()

replacements = {
    # ProbabilityPanel
    '"Ereignis:"': '"Event:"',
    '"Live (beim Schieben sofort rechnen)"': '"Live (compute while sliding)"',
    '"Berechnen"': '"Compute"',
    '"Ergebnis:"': '"Result:"',
    
    # QuantilePanel
    '"q  (Wahrscheinlichkeit):"': '"q (Probability):"',
    '"Quantil berechnen"': '"Compute Quantile"',
    '"→ kleinstes x mit P(X ≤ x) ≥ q"': '"→ smallest x with P(X ≤ x) ≥ q"',
    
    # SamplePanel
    '"Stichprobengröße n:"': '"Sample size n:"',
    '"Seed (0 = zufällig):"': '"Seed (0 = random):"',
    '"zufällig"': '"random"',
    '"Ziehen"': '"Draw Sample"',
    '"Mittelwert / SD / Min / Max:"': '"Mean / SD / Min / Max:"',
    
    # SolvePanel
    '"Unbekannter Parameter:"': '"Unknown Parameter:"',
    '"Suchbereich min:"': '"Search Range min:"',
    '"Suchbereich max:"': '"Search Range max:"',
    '"Parameter lösen"': '"Solve Parameter"',
    '"(keine Parameter)"': '"(no parameters)"',
    '"Keine Lösung im Suchbereich gefunden."': '"No solution found in search range."',
    '"Suchbereich enthält ungültige Werte — Grenzen anpassen."': '"Search range contains invalid values — adjust bounds."',
    '"Lösung liegt nicht im Suchbereich (Vorzeichen identisch). "': '"Solution not in search range (identical signs). "',
    '"Grenzen erweitern."': '"Expand bounds."',
    
    # MainWindow
    '"StatiQ — Statistik Studio"': '"StatiQ — Statistics Studio"',
    '"Verteilungen filtern..."': '"Filter distributions..."',
    'f"Diskret ({len(DISCRETE)})"': 'f"Discrete ({len(DISCRETE)})"',
    'f"Kontinuierlich ({len(CONTINUOUS)})"': 'f"Continuous ({len(CONTINUOUS)})"',
    '"Parameter"': '"Parameters"',
    '"Operation"': '"Operations"',
    '"Wahrscheinlichkeit"': '"Probability"',
    '"Quantil"': '"Quantile"',
    '"Zufallsstichprobe"': '"Random Sample"',
    '"Plot exportieren (PNG)"': '"Export Plot (PNG)"',
    '"&Datei"': '"&File"',
    '"Beenden"': '"Exit"',
    '"&Hilfe"': '"&Help"',
    '"Über"': '"About"',
    '"Über StatiQ"': '"About StatiQ"',
    '"Interaktives Tool für Wahrscheinlichkeiten, Quantile und Stichproben.\\n"': '"Interactive tool for probabilities, quantiles, and sampling.\\n"',
    '"Basiert auf PyQRS, komplett überarbeitet für bessere Zugänglichkeit."': '"Based on PyQRS, completely overhauled for learning and accessibility."',
    '"Plot speichern"': '"Save Plot"',
    '"Plot erfolgreich unter {path} gespeichert."': 'f"Plot successfully saved to {path}."',
    '"Fehler"': '"Error"',
    
    # Category lists
    '"diskret"': '"discrete"',
    '"kontinuierlich"': '"continuous"',
}

for k, v in replacements.items():
    text = text.replace(k, v)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Done translating main.py")
