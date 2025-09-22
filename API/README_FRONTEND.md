# Grammar AI - Entity Recognition Frontend

Un frontend simplu pentru recunoașterea entităților în text folosind spaCy și Flask.

## Caracteristici

- 🧠 Recunoaștere automată a entităților în text (persoane, organizații, locații, date, etc.)
- 🎨 Highlight vizual al entităților cu culori diferite
- 📊 Sumar organizat pe categorii
- 💻 Interfață modernă și responsive
- ⚡ API rapid bazat pe Flask

## Instalare

### 1. Instalează dependențele Python

```bash
pip install -r requirements.txt
```

### 2. Descarcă modelul spaCy

```bash
python -m spacy download en_core_web_sm
```

### 3. Pornește aplicația

```bash
python app.py
```

### 4. Deschide browserul

Navighează la: http://localhost:5000

## Utilizare

1. **Introdu textul** în zona de text din partea de sus
2. **Apasă "Analizează Text"** pentru a procesea textul
3. **Vizualizează rezultatele**:
   - Textul cu entitățile evidențiate colorat
   - Sumarul entităților organizate pe categorii

## Tipuri de entități recunoscute

- **🟣 Persoane** (PERSON) - nume de persoane
- **🔵 Organizații** (ORG) - companii, instituții
- **🟢 Locații** (GPE/LOC) - țări, orașe, locuri
- **🟠 Date** (DATE) - date și perioade
- **🟡 Ore** (TIME) - timp și ore
- **💛 Bani** (MONEY) - sume de bani
- **🟤 Cantități** (CARDINAL/QUANTITY) - numere și cantități
- **⚫ Altele** - alte tipuri de entități

## Structura proiectului

```
GrammarAI/
├── app.py              # Aplicația Flask principală
├── Entities.py         # Codul original de extragere entități
├── templates/
│   └── index.html      # Template-ul HTML principal
├── static/
│   ├── style.css       # Stilurile CSS
│   └── script.js       # Logica JavaScript
├── requirements.txt    # Dependențele Python
└── README_FRONTEND.md  # Acest fișier
```

## API Endpoints

### POST /analyze
Analizează un text și returnează entitățile găsite.

**Request:**
```json
{
    "text": "Sarah Johnson lucrează la Microsoft în New York."
}
```

**Response:**
```json
{
    "text": "Sarah Johnson lucrează la Microsoft în New York.",
    "entities": [
        {
            "text": "Sarah Johnson",
            "label": "PERSON",
            "description": "People, including fictional",
            "start": 0,
            "end": 13
        }
    ],
    "categorized": {
        "PEOPLE": [...],
        "ORGANIZATIONS": [...]
    },
    "total_entities": 3,
    "model_used": "en_core_web_sm"
}
```

### GET /health
Verifică starea aplicației.

## Probleme comune

### Model spaCy lipsă
```
Error loading spaCy model: No spaCy model available
```
**Soluție:** Rulează `python -m spacy download en_core_web_sm`

### Port ocupat
```
Address already in use
```
**Soluție:** Modifică portul în `app.py` sau oprește aplicația care folosește portul 5000.

## Personalizare

### Adăugare de noi tipuri de entități
Modifică mapping-urile din `script.js` și `app.py` pentru a adăuga suport pentru noi tipuri.

### Schimbarea culorilor
Editează clasele CSS din `static/style.css` pentru a personaliza culorile highlight-ului.

### Îmbunătățirea modelului
Înlocuiește `en_core_web_sm` cu `en_core_web_md`, `en_core_web_lg` sau `en_core_web_trf` pentru o precizie mai bună.

## Licență

Acest proiect este licențiat sub MIT License.
