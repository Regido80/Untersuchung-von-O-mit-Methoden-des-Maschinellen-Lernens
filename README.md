# Untersuchung-von-O-mit-Methoden-des-Maschinellen-Lernens

In den letzten Jahren gewannen die Begriffe Künstliche Intelligenz, Neuronale Netze und Deep Learning immer weiter an medialer Aufmerksamkeit/Bedeutung.
Meldungen über neue Möglichkeiten etwa in der Gesichtserkennung,dem autonomen Fahren oder der Simulation menschlichen Verhaltens erzeugen grosses Interesse und ein weites gesellschaftliches Echo.
Dennoch sind KI Werkzeuge nicht allmächtig und kognitive Fähigkeiten wie selbstständiges Denken sind noch nicht maschinell nachahmbar.
Damit bleiben auch wesentliche Teile der höheren Mathematik dem Zugang und der Bearbeitung durch Neuronale Netze verschlossen.

Im Folgenden soll eine Möglichkeit vorgestellt werden wie, auf KI basierende Klassifikationsverfahren und Vergleichsmethoden, auf gewisse mathematische Objekte, den sogenannten Origamis, angewandt werden können.

# 1 Einführung

## 1.1 Origamis

Origamis sind mathematische Objekte die sich recht anschaulich definieren lassen:

**Definition**:

Ein Origami O ist, bis auf Äquivalenz, definiert durch eine geschlossene Fläche, die gegeben ist durch Verkleben von endlich vielen Einheitsquadraten nach den sogenannten 

Origami-Regeln:

• jede obere Kante wird mit einer eindeutig bestimmten unteren Kante und

• jede rechte Kante wird mit einer eindeutig bestimmten linken Kante verklebt


Ein einfaches Beispiel ist der Torus (Donut), der durch Verkleben gegenüberliegende Seiten eines Einheitsquadrates entsteht.

Diese Definition erlaubt es Origamis auch rein kombinatorisch durch zwei Permutationen sigma_ x und sigma_y in S_d, zu fassen, wobei

• sigma_x die horizontale und

• sigma_y die vertikale 

Verklebestruktur beschreibt.


Obwohl damit Origamis kombinatorisch einfach zu fassende mathematische Objekte sind, tragen sie tiefliegende mathematische Strukturen und bilden somit einen Zugang für Methoden des maschinellen Lernens zur höheren Mathematik.

Diese und weitere Informationen zu Origamis sind nachzulesen z.B. in:

F. Herrlich und G. Schmithüsen, Dessin d’Enfants and Origami curves, A.
Papadopoulos (ed.) Handbook of Teichmüller theory, Vol. II, chapter 18.
European Mathematical Society (2009)

## 1.2 Künstliche Neuronale Netze und Deep Learning

Künstliche Neuronale Netze (KNN) sind EDV-Systeme die aus verbundenen Schichten künstlicher Neuronen zusammengesetzt sind.

Die Architektur orientiert sich hierbei an biologischen Vorlagen:

Einer Eingabeschicht werden die zu verarbeitenden Daten übergeben, um dann gewichtet und durch eine Schwellenfunktion geregelt über Zwischenschichten (Hidden Layer) verarbeitet zu werden und an einer Ausgabeschicht ein Ergebnis zu liefern.

Tiefe Neuronale Netze (TNN) weisen eine vergleichsweise große Anzahl von sogenannten Hidden Layers zwischen Eingabe- und Ausgabeschicht auf.

### 1.2.1 Clusteranalyse

Unter dem Begriff Clusteranalyse werden maschinelle Verfahren zur Gruppierung und Strukturierung von Objekten (Daten) nach gewissen Merkmalen der Ähnlichkeit oder Nähe in einem Eigenschaftsraum zusammengefasst.

Im Gegensatz zu Klassifizierungsverfahren werden hier keine gelabelten Trainingsdaten zum Erlernen eines Zuordnungsmodells benötigt.

Man unterscheidet im Wesentlichen drei Kategorien:

    •     Partitions-Clustering
    
    •     Dichtebasiertes Clustering
    
    •     Hierarchisches Clustering


Im Folgenden wird aus jeder dieser Kategorien jeweils ein Clusteranalyse Algorithmus vorgestellt:

#### **K-Means Verfahren**:

K-Means ist ein Partitions-Clustering Verfahren.

Algorithmus:

    1. Manuelles Festlegen der Clusteranzahl und zufälliges Festlegen der anfänglichen Clusterzentren
    
    2. Ermitteln der (quadrierten euklidischen) Abstände der Datenpunkte zu den Clusterzentren und anschließende Zuordnung zum nächstgelegenen Zentrum
    
    3. Neuberechnung der Clusterzentren durch Mittelwertbildung aller Abstände zwischen Datenpunkten eine Clusters
    
    4. Wiederholen dieser Schritte für eine festgelegte Anzahl von Iterationen oder bis sich die Gruppenzentren zwischen den Iterationen nicht mehr wesentlich ändern. 

Vorteile:

    • Algorithmus mit linearer Komplexität O(n)

Nachteile:

    • Manuelle Auswahl der Anzahl der Cluster
    
    • Zufällige Initiierung der Clusterzentren liefert bei verschiedenen Durchläufen u.U. unterschiedliche, also nicht wiederholbare Ergebnisse

#### **Dichtebasiertes räumliches Clustering von Anwendungen mit Rauschen (DBSCAN)**

DBSCAN ist ein dichtebasierter Cluster-Algorithmus

Algorithmus:

    1. Wähle einen beliebigen (nicht als “abgearbeitet” markierten Punkt als) Start Datenpunkt eines neuen Clusters und kennzeichne ihn als Kandidaten.
    
    2. Kennzeichne den Kandidaten als aktuellen Punkt, falls eine Mindestanzahl von Datenpunkten (Nachbarschafts Punkte) in einer vorgegebenen 𝜺-Umgebung um ihn liegen (Mindestdichte) und markiere ihn als abgearbeitet.
       Ansonsten kennzeichne den Kandidaten als Rauschen, markiere ihn als abgearbeitet und wähle einen neuen (nicht als “abgearbeitet” markierte) Datenpunkt als Kandidaten (beliebig falls Kandidat Start Datenpunkt war, ansonsten aus der ε-Umgebung des aktuellen Punktes).

    3. Ordne alle Punkte die innerhalb der ε-Umgebung des aktuellen Punktes liegen dem aktuellen Cluster zu.
    
    4. Wähle einen nicht als “abgearbeitet” markierten Punkt aus der ε-Umgebung des aktuellen Punktes als neuen Kandidaten.
    
    5. Wiederhole Schritte 2, 3 und 4 bis keine weiteren Punkte dem Cluster hinzugefügt werden, d.h. alle Punkte des Clusters als abgearbeitet markiert wurden.
    
    6. Wiederhole die Schritte 1 bis 5, bis alle Datenpunkte als abgearbeitet markiert wurden

Vorteile:

    • Keine manuelle Vorgabe der Clusteranzahl
    
    • Identifiziert Ausreißer als Rauschen
    
    • Findet zuverlässig beliebig große und beliebig geformte Cluster

Nachteile:

    • Funktioniert schlecht bei variierender Clusterdichte (Abstandsschwelle ε und minPoints)
    
    
#### **Agglomeratives hierarchisches Clustering**

Hierarchische Clustering-Algorithmen lassen sich in zwei Kategorien einteilen:
Bottom Up und Top Down.

Bottom-up-Algorithmen fassen zu Beginn jeden Datenpunkt als einen einzelnen Cluster auf und führen diese dann schrittweise zusammen (agglomerieren sie), bis schließlich alle Datenpunkte in einem einzigen Cluster enthalten sind.

Sie werden meist als Baumdiagramm (oder Dendrogramm) dargestellt.


Algorithmus:

    1. Fasse jeden Datenpunkt als einzelnen Cluster auf.
       Wähle eine Abstandsmetrik, die den Abstand zwischen zwei Clustern misst.
       (Meist durchschnittlichen Abstand zwischen Datenpunkten im ersten Cluster und Datenpunkten im zweiten Cluster)
       
    2. Kombiniere je zwei Cluster, die, nach einer festgelegten Abstandsfunktion (etwa durchschnittlichen Entfernung zwischen Datenpunkten im ersten Cluster und Datenpunkten im zweiten Cluster), den geringsten Abstand untereinander haben.
    
    3. Wiederhole diesen Schritt bis alle Datenpunkte in einem Cluster zusammengefasst sind.

Vorteile:

    • Keine manuelle Vorgabe der Clusteranzahl
    
    • Liefert in einem Durchlauf Clusteringergebnisse mit unterschiedlichen Clusteranzahlen
    
    • Reagiert nicht bei Änderung der Wahl der Entfernungsmetrik
    
    • Erkennt hierarchische Strukturen in den Daten

Nachteile:

    • Geringe Effizienz (zeitliche Komplexität von O (n³))
    
#### **Autoencoder**

Ein wesentliches Problem bei der Anwendung von Clusteringmethoden ist die Dimension des Eigenschaftraums (Feature Space), in dem die zu verarbeitenden Daten strukturiert werden. Bei der Bildverarbeitung zum Beispiel entspricht diese Dimension der Pixelanzahl.
Je höher diese Dimension ist, desto schlechter arbeitet der Clusteringalgorithmus.

Um dieses Problem zu umgehen können sogenannte Autoencoder verwendet werden.

Das sind zweistufige Algorithmen, die im ersten Schritt, dem Encoding, Eingabedaten auf Merkmale mit niedrigerer Dimension komprimieren, um sie im zweiten Schritt, dem Decoding, aus den komprimierten Daten wieder zu rekonstruieren. 

Die zugehörige (meist spiegelsymmetrische) Implementierung als Neuronales Netz wird trainiert in dem die rekonstruierten Daten mit den zugehörigen Eingaben verglichen werden und das KNN anschließend entsprechend angepasst wird.

Die Clusteringalgorithmen werden dann auf die niedrig dimensionalen encodierten Daten angewendet.

### 1.2.2 One Shot Learning

Ein wesentliches Problem bei Klassifizierungsverfahren ist, dass sie viele gelabelte Trainingsdaten (Daten, die mit ihrer Klassifizierungszuordnung annotiert sind) erfordern. In vielen Anwendungen ist es manchmal nicht möglich, so viele Daten zu sammeln. One Shot Learning soll dieses Problem lösen.

Im Folgenden soll ein Verfahren dazu vorgestellt werden:

One Shot Learning mit Siamesischen Netzwerken

Ein Siamesisches Neuronales Netzwerk besteht aus zwei, in Architektur und Gewichten identischen Netzwerken, die parallel unabhängige Eingaben verarbeiten und anhand eines erlernten Modells einen Ähnlichkeitswert etwa zwischen 0 und 1 
ausgeben, wobei die Eins im Fall identischer Eingaben geliefert wird / liefern, der die Wahrscheinlichkeit angibt, dass beiden Eingaben identisch sind.

Ein solches Netzwerk lernt also nicht eine Eingabe direkt einer der Ausgabeklassen zuzuordnen. Vielmehr lernt es eine Ähnlichkeitsfunktion, die zwei Eingaben vergleicht und ausdrückt, wie ähnlich sie sind.

Die verwendete Netzwerkarchitektur und die Hyperparameter folgen hierbei der Methodik, die im Paper 

**Siamese Neural Networks for One-shot Image Recognition**

von Gregory Koch, Richard Zemel und Ruslan Salakhutdinov beschrieben werden.

Insbesondere werden dabei Convolutional Neuronal Networks (CNN) verwendet, die lokalisierte Merkmale aus Eingangsbildern extrahieren und diese Bildfelder mittels Filtern auffalten.

# 1.3 Darstellung von Origamis als Pixelmuster in Python

Im Folgenden sollen alle Algorithmen vorgestellt werden die zur Erzeugung und Darstellung aller Origamis einer bestimmten Länge in Python verwendet werden.

Dabei wird in folgender Reihenfolge vorgegangen:

    1. Erzeugen aller Zykeltypen der Länge l /in /N (horizontale Verklebung)
       Übersetze in Standarddarstellung
       
    2. Erzeuge alle Permutationen der Länge l /in /N (vertikale Verklebung)
       Übersetze in Zykeldarstellung
       
    3. Erzeuge alle Origamis der Länge l /in /N
    
    4. Sortiere alle nicht zusammenhängenden Origamis aus
    
    5. Erzeuge zugehörige Pixelmuster

### 1.3.1 Erzeuge alle Origamis der Länge l /in /N 

#### **Erzeugen aller Zykeltypen der Länge l /in /N**

Die Verklebestruktur eines Origamis ist invariant unter Umnummerierung der Einheitsquadrate.
Damit ist die gesamte Information über die Verklebung in eine Richtung bereits durch die Zykelstruktur festgelegt.

Die Bestimmung der Zykeltypen der Länge l /in /N ist dabei mathematisch identisch zu der Bestimmung der Summandenzerlegung von l.
      
Erzeuge zunächst die Liste L der Zykellängen einer Permutation der Länge l. Aus ihr werden in folgender Weise alle entsprechenden Zykeltypen erzeugt:

    • Jeder Zykeltyp wird erzeugt als Liste von Listen, wobei jede Unterliste (Zykelliste) einem Zykel entspricht und deren Längen durch den zugehörigen Eintrag in L bestimmt ist.
    
    • Die Zykellisten sind nach der Länge geordnet, von kürzeste zu längste.
    
    • Die Elemente der Zykellisten sind genau die Zahlen von 1 bis l, die aufsteigend, beginnend mit der Eins und über alle Listen hinweg fortlaufend eingesetzt werden. 

Diese Vorgehensweise sichert eine eindeutige und konsistente Darstellung der Zykeltypen.

#### **Erzeugen aller Permutation der Länge l /in /N**

Eine Liste aller Permutationen wird in Python als Liste von Tupeln durch die Funktion

	itertools.permutations()

erzeugt.

#### **Erzeugen aller Origamis der Länge l /in /N**

Kombiniere je einen

    • Zykeltyp 	in Zykel- und Standarddarstellung   (horizontale Verklebung)  mit einer
    
    • Permutation 	in Zykel- und Standarddarstellung   (vertikale Verklebung)

#### **Aussortieren aller nicht zusammenhängender Origamis**

Abgeschlossene Verklebungen von Einheitsquadraten nach Vorgabe eines Zykels werden im Folgenden als Block bezeichnet.

Um zu überprüfen, ob ein Origami zusammenhängend ist, genügt es festzustellen, ob etwa ein Teil der horizontalen Blöcke vertikal (quadrat weise) nur mit sich selbst verklebt sind.

Es müssen dabei nicht alle (horizontalen) Blockkombinationen überprüft werden, denn falls eine Blockkombination bzgl. vertikaler Verklebung nicht abgeschlossen ist, so ist auch dessen Komplement nicht abgeschlossen.

### 1.3.2 Erzeugen von Pixelmustern für alle Origamis der Länge l /in /N

In diesem Abschnitt sollen zwei verschiedene Pixeldarstellungen für Origamis zur Übergabe an Neuronale Netze vorgestellt werden.

Diese orientieren sich an prominenten Datensätzen aus dem Bereich Computer Vision (Deep Learning Verfahren zur Bilderkennung).

Darüber hinaus sind eine Vielzahl anderer Darstellungen denkbar.

Die Pixelmuster können auch augmentiert werden etwa durch weitere zahlenmäßig/kombinatorisch darstellbare/kodierbare Eigenschaften.

Es ist auch möglich einzelne Merkmale mehr oder weniger zu betonen, so dass sie bei der Verarbeitung durch Neuronale Netze unterschiedlich stark berücksichtigt werden.

#### **Pixeldarstellung in Anlehnung an den MNIST, Fashion MNIST Datensatz**

Der Fashion-MNIST Datensatz besteht aus gelabelten graustufen Artikelbildern des Versandhändlers Zalando vom Format 28x28 (784 Pixel), die jeweils mit einer Beschriftung aus 10 Klassen versehen sind.
Die Daten setzten sich dabei zusammen aus 60 000 Trainings- und 10 000 Test Beispielen, wobei jede Klasse gleich häufig auftritt.
Jedem Pixel ist ein einzelner Pixelwert zugeordnet, der die Graustufe als Wert zwischen 0 und 255 angibt, wobei höhere Zahlen dunkleren Pixeln entsprechen. 
Die Trainings- und Testdatensätze haben 785 Spalten, wobei die erste Spalte die Klassenbezeichnungen (siehe oben) enthält. Der Rest der Spalten enthält die Pixelwerte des zugeordneten Bildes.

Algorithmus zur Erzeugung des Pixelmusters:

    • Erzeuge eine Pixelmatrix vom Format (2*l)x(l+1)
    
    • Beginnend beim l-ten Pixel in der obersten Zeile und nach links fortfahrend, setze als Pixelwert den Wert des entsprechenden Eintrags der Standarddarstellung der horizontalen Verklebestruktur
    
    • Beginnend beim (l+1)-ten Pixel in der obersten Zeile und nach rechts fortfahrend, setze als Pixelwert den Wert des entsprechenden Eintrags der Standarddarstellung der vertikalen Verklebestruktur
    
    • Beginnend beim l-ten Pixel in der zweitobersten Zeile und nach links bzw. bei Zykelwechsel nach unten fortfahrend, setze als Pixelwert den Wert des entsprechenden Eintrags des aktuellen Zykels, beginnend mit dem Ersten, der Zykeldarstellung der horizontalen Verklebestruktur
   
    • Beginnend beim (l+1)-ten Pixel in der zweitobersten Zeile und nach rechts bzw. bei Zykelwechsel nach unten fortfahrend, setze als Pixelwert den Wert des entsprechenden Eintrags des aktuellen Zykels, beginnend mit dem Ersten, der Zykeldarstellung der vertikalen Verklebestruktur
 
#### **Pixeldarstellung in Anlehnung an den Omniglot Datensatz**

Der Omniglot-Datensatz besteht aus graustufen Bildern im Format 105x105 von handgeschriebenen Zeichen nach Vorlage von insgesamt 1623 Buchstaben aus 50 verschiedenen Alphabeten.
Für jedes Zeichen gibt es nur 20 Beispiele die jeweils von einer anderen Person geschrieben wurden.
Jedem Pixel ist ein einzelner Pixelwert zugeordnet, der die Graustufe als Wert zwischen 0 und 255 angibt, wobei höhere Zahlen dunkleren Pixeln entsprechen.

Der Datensatz kann unter

	GitHub - brendenlake/omniglot: Omniglot data set for one-shot learning
heruntergeladen werden.


Erzeugung der Pixelmusters:

	M := {Menge aller Einträge i.d. bisher abgearbeiteten horizontalen Zykeln}
  
	N := {Menge aller Einträge i.d. bisher abgearbeiteten vertikalen Zykeln}

Vorgehensweise:

    • Ordne jedem horizontalen Zykeleintrag ein Pixelblock vom Format
	      2a x a
	    zu
      
    • Ordne jedem vertikalen Zykeleintrag ein Pixelblock vom Format
	      a x 2a
	    zu
      
    • Lasse einen Rand/Rahmen mit einer Stärke von 2a Pixeln/Kästchen
    
Algorithmus zur Erzeugung des Pixelmusters:

    Horizontaler Anfangsblock:
    
    • Beginne bei der Anfangskoordinate (53, 53)
    
    • Erzeuge einen horizontalen Block nach links nach Vorgabe des ersten horizontalen Zykels
    
    • Aktualisiere M
    
    • Wähle den vertikalen Zykel, der den letzten Eintrag des aktuellen horizontalen Zykels enthält als aktuellen vertikalen Zykel

    Vertikaler Ansatzblock:
    
    • Wähle als Ansatzkoordinaten den letzten (a x a)-Pixelblock des letzten Eintrags des horiziontalen Anfangsblockss
    
    • Setze einen vertikalen Block nach Vorgabe des aktuellen vertikalen Zykels nach oben an, falls der Zahlwert des Ansatzeintrags gerade ist andernfalls nach unten
   
    • Aktualisiere N
   
    • Überprüfe jeden Eintrag des aktuellen vertikalen Zykels, beginnend mit dem letzten, ob dieser bereits in M liegt. Ist dies der Fall, überprüfe den nächsten Eintrag. Ist dies nicht der Fall (und der Eintrag ist horizontal nicht mit sich selbst verklebt), wähle diesen als Ansatzeintrag und denjenigen horizontalen Zykel der diesen Eintrag enthält als aktuellen horizontalen Zykel 

    Horizontaler Ansatzblock:
    
    • Wähle als Ansatzkoordinaten den letzten (a x a)-Pixelblock des Ansatzeintrags des aktuellen vertikalen Zykels
   
    • Setze einen horizontalen Block nach Vorgabe des aktuellen horizontalen Zykels nach links an, falls der Zahlwert des Ansatzeintrags gerade ist andernfalls nach rechts
    
    • Aktualisiere M
    
    • Überprüfe jeden Eintrag des aktuellen horizontalen Zykels, beginnend mit dem letzten, ob dieser bereits in N liegt. Ist dies der Fall, überprüfe den nächsten Eintrag. Ist dies nicht der Fall (und der Eintrag ist vertikal nicht mit sich selbst verklebt), wähle diesen als Ansatzeintrag und denjenigen vertikalen Zykel der diesen Eintrag enthält als aktuellen vertikalen Zykel. 

    Wiederhole die letzten beiden Schritte, bis gilt
	    M = {1, 2, …, l}

Wiederhole anschließend obigen Algorithmus, wobei zunächst ein vertikaler Anfangsblock nach rechts erzeugt wird und die Fortsetzungsrichtungen in Abhängigkeit vom Wert des Ansatzeintrags vertauscht sind.

## 2 Anwendung und Ergebnisse

## 2.1 Anwendung von Clusteringverfahren auf Origamis

### 2.1.1 K-Means, DBSCAN, Agglomerative Clustering, ohne Autoencoder

Ohne die Anwendung eines Autoencoders (zur Reduktion der Dimensionen des Eigenschaftsraum) liefern die Clusteringverfahren keine brauchbaren Ergebnisse.

### 2.1.2 K-Means, DBSCAN, Agglomerative Clustering, mit Autoencoder

Die auf die encodierten Daten angewendeten Clusteringalgorithmen liefern brauchbare Ergebnisse, wobei unter den verwendeten Autoencoder Architekturen der Deep Autoencoder (ohne Convolutional Layer) am Besten arbeitet. 

K-Means zeigt die Tendenz vornehmlich nach der horizontalen Verklebestruktur zu Clustern.

In jedem Fall ist eine Augmentierung und Ausprobieren weiterer Darstellungen der Daten sinnvoll.

Der Code für die Autoencoder wurde im Wesentlichen aus

https://blog.keras.io/building-autoencoders-in-keras.html
  
übernommen und entsprechend angepasst.

## 2.2 Anwendung von One-Shot-Learning auf Origamis zum Auffinden ähnlicher Origamis

Da das Siamese Network direkt auf der Liste der Pixeldarstellungen trainiert und es je Pixeldarstellung/Origami nur ein Beispiel gibt, ist die Gefahr für Overfitting recht hoch.

Grundsätzlich sind folgende Alternativen sinnvoll:

    • Erzeugen weiterer Beispiele für die Pixeldarstellung eines Origamis durch Variation der angesetzten horizontalen/vertikalen Blöcke (z.B. Blockbreite-, dicke, leicht schräge Blöcke etc.) und des jeweiligen Ansatzblocks (pixelweises Verschieben)

    • Trainieren des Modells auf dem Omniglot Datensatz (zur Feature Extraction) und anschließendes Anwenden auf die Liste der Pixeldarstellungen

Auch hier ist eine Augmentierung oder Erweiterung der Pixeldarstellung empfehlenswert.

Der Code für das Siamese Network wurde im Wesentlichen aus

https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks/blob/master/Siamese%20on%20Omniglot%20Dataset.ipynb

übernommen und entsprechend angepasst.

