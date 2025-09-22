# SpaCy NER pentru Email-uri - Doar Text
# Extrage și procesează doar textul din email-uri

import spacy
import pandas as pd
from collections import defaultdict
import json
import os
import kagglehub

# Funcție pentru încărcarea celui mai bun model disponibil
def load_best_available_model():
    """Încarcă cel mai bun model disponibil în Kaggle"""
    
    models_to_try = [
        "en_core_web_trf",   # Model transformer - cel mai bun
        "en_core_web_lg",    # Model mare
        "en_core_web_md",    # Model mediu  
        "en_core_web_sm"     # Model mic - ultimă opțiune
    ]
    
    for model_name in models_to_try:
        try:
            nlp = spacy.load(model_name)
            print(f"✓ Model încărcat cu succes: {model_name}")
            return nlp, model_name
        except OSError:
            print(f"✗ {model_name} nu este disponibil")
            continue
    
    # Dacă niciun model nu e găsit, instalează unul mic
    print("Nu s-a găsit niciun model. Se instalează en_core_web_sm...")
    import subprocess
    import sys
    
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    return nlp, "en_core_web_sm"

# Încarcă modelul
nlp, model_name = load_best_available_model()
print(f"\nSe folosește modelul: {model_name}")
print(f"Pipeline: {nlp.pipe_names}")

class PhishingTextNER:
    """Extragere NER simplă pentru texte de phishing"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
    
    def extract_entities(self, text):
        """Extrage entități din text folosind spaCy"""
        try:
            # Procesează textul
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            print(f"Eroare la procesarea textului: {e}")
            return []
    
    def find_urgency_keywords(self, text):
        """Găsește cuvinte cheie care indică URGENCY în text"""
        urgency_keywords = [
            # Cuvinte simple
            "urgent", "asap", "immediately", "emergency", "critical", "rush", 
            "priority", "now", "quickly", "fast", "deadline", "expires", 
            "expiring", "soon", "today", "tomorrow", "hurry", "hurry up",
            "instantly", "promptly", "swiftly", "rapidly", "overdue", "expired",
            "last chance", "final notice", "act immediately", "respond immediately",
            "take action", "urgent action", "critical issue", "emergency situation",
            "rush order", "priority task", "high priority", "top priority",
            "maximum priority", "urgent request", "immediate response", "quick response",
            "fast response", "urgent matter", "critical matter", "emergency matter",
            "urgent issue", "critical issue", "emergency issue", "urgent problem",
            "critical problem", "emergency problem", "action required", "security alert",
            "warning", "attention", "act fast", "click here", "verify now", "reset now",
            "renew now", "confirm now", "as soon as possible", "right away",
            "time-sensitive", "time sensitive", "24 hours", "immediate", "quick",
            "urgent", "critical", "emergency", "priority", "rush", "asap", "now",
            "immediately", "fast", "quickly", "hurry", "deadline", "expires",
            "expiring", "soon", "today", "tomorrow", "overdue", "expired",
            "last chance", "final notice", "urgent!", "critical!", "emergency!",
            "priority!", "rush!", "asap!", "now!", "immediately!", "fast!",
            "quickly!", "hurry!", "deadline!", "expires!", "expiring!", "soon!",
            "today!", "tomorrow!", "overdue!", "expired!", "last chance!",
            "final notice!", "urgent:", "critical:", "emergency:", "priority:",
            "rush:", "asap:", "now:", "immediately:", "fast:", "quickly:",
            "hurry:", "deadline:", "expires:", "expiring:", "soon:", "today:",
            "tomorrow:", "overdue:", "expired:", "last chance:", "final notice:"
        ]
        
        found_urgencies = []
        text_lower = text.lower()
        
        for keyword in urgency_keywords:
            keyword_lower = keyword.lower()
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                end = pos + len(keyword)
                found_urgencies.append({
                    'text': text[pos:end],
                    'label': 'URGENCY',
                    'start': pos,
                    'end': end
                })
                start = pos + 1
        
        return found_urgencies
    
    def merge_entities_without_overlap(self, spacy_entities, urgency_entities):
        """Combină entitățile spaCy cu cele URGENCY fără suprapuneri"""
        all_entities = []
        used_positions = set()
        
        # Adaugă entitățile spaCy existente
        for entity in spacy_entities:
            all_entities.append(entity)
            # Marchează pozițiile ca folosite
            for pos in range(entity['start'], entity['end']):
                used_positions.add(pos)
        
        # Adaugă entitățile URGENCY doar dacă nu se suprapun
        for urgency_entity in urgency_entities:
            overlap = False
            for pos in range(urgency_entity['start'], urgency_entity['end']):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                all_entities.append(urgency_entity)
                # Marchează pozițiile ca folosite
                for pos in range(urgency_entity['start'], urgency_entity['end']):
                    used_positions.add(pos)
        
        # Sortează entitățile după poziția de start
        all_entities.sort(key=lambda x: x['start'])
        
        return all_entities
    
    def process_text_only(self, text):
        """Procesează doar textul și returnează rezultatele"""
        # Curăță textul
        clean_text = text.strip()
        
        # Extrage entitățile cu spaCy
        spacy_entities = self.extract_entities(clean_text)
        
        # Găsește cuvinte cheie URGENCY
        urgency_entities = self.find_urgency_keywords(clean_text)
        
        # Combină entitățile fără suprapuneri
        all_entities = self.merge_entities_without_overlap(spacy_entities, urgency_entities)
        
        # Formatează rezultatul ca tuple (text, {"entities": [(start, end, label)]})
        entity_tuples = [(entity['start'], entity['end'], entity['label']) for entity in all_entities]
        formatted_result = (clean_text, {"entities": entity_tuples})
        
        return {
            'clean_text': clean_text,
            'entities': all_entities,
            'spacy_entities': spacy_entities,
            'urgency_entities': urgency_entities,
            'formatted_result': formatted_result
        }

# Funcție pentru încărcarea textelor de phishing
def load_phishing_texts():
    """Încarcă textele de phishing din dataset"""
    try:
        print("Încărcare dataset phishing...")
        
        # Metoda nouă de încărcare kagglehub (fără deprecated load_dataset)
        path = kagglehub.dataset_download("ahmadtijjani/phishing-urgency-authority-persuasion")
        print(f"✓ Dataset descărcat la: {path}")
        
        # Găsește fișierele din dataset
        import glob
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        json_files = glob.glob(os.path.join(path, "*.json"))
        
        print(f"Fișiere găsite:")
        print(f"  CSV: {csv_files}")
        print(f"  Parquet: {parquet_files}")
        print(f"  JSON: {json_files}")
        
        df = None
        
        # Încearcă să încărce primul fișier disponibil
        if csv_files:
            print(f"Încărcare CSV: {csv_files[0]}")
            df = pd.read_csv(csv_files[0])
        elif parquet_files:
            print(f"Încărcare Parquet: {parquet_files[0]}")
            df = pd.read_parquet(parquet_files[0])
        elif json_files:
            print(f"Încărcare JSON: {json_files[0]}")
            df = pd.read_json(json_files[0])
        else:
            # Caută recursive în subdirectoare
            all_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.csv', '.parquet', '.json')):
                        all_files.append(os.path.join(root, file))
            
            if all_files:
                file_path = all_files[0]
                print(f"Încărcare fișier găsit: {file_path}")
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
        
        if df is None:
            print("✗ Nu s-a putut încărca niciun fișier")
            return []
        
        print(f"✓ Dataset încărcat cu {len(df)} rânduri")
        print(f"✓ Coloane disponibile: {list(df.columns)}")
        print("Primele 3 înregistrări:")
        print(df.head(3))
        
        # Verifică dacă coloana 'text' există
        if 'text' not in df.columns:
            print("✗ Coloana 'text' nu există în dataset")
            print(f"Coloane disponibile: {list(df.columns)}")
            
            # Încearcă să găsească o coloană care pare să conțină text
            text_column = None
            possible_columns = ['message', 'content', 'body', 'email', 'phishing_text', 'sample']
            
            for col in possible_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column:
                print(f"✓ Se folosește coloana '{text_column}' în loc de 'text'")
                phishing_texts = df[text_column].dropna().tolist()
            else:
                # Folosește prima coloană care pare să conțină text lung
                for col in df.columns:
                    if df[col].dtype == 'object':
                        avg_length = df[col].dropna().astype(str).str.len().mean()
                        if avg_length > 20:
                            text_column = col
                            break
                
                if text_column:
                    print(f"✓ Se folosește coloana '{text_column}' (detectată automat)")
                    phishing_texts = df[text_column].dropna().tolist()
                else:
                    return []
        else:
            print(f"✓ Se folosește coloana 'text' pentru mesajele de phishing")
            phishing_texts = df['text'].dropna().tolist()
        
        # Filtrează textele care sunt prea scurte
        phishing_texts = [text for text in phishing_texts if len(str(text).strip()) > 10]
        
        # Process all texts, no limit
        # phishing_texts = phishing_texts[:150]  # Removed limit
        
        print(f"✓ Încărcate {len(phishing_texts)} texte de phishing pentru procesare")
        
        # Afișează câteva exemple
        print(f"\nExemple de texte încărcate:")
        for i, text in enumerate(phishing_texts[:3], 1):
            print(f"{i}. {str(text)[:80]}...")
        
        return phishing_texts
        
    except Exception as e:
        print(f"✗ Eroare la încărcarea dataset-ului: {e}")
        print("Se vor folosi exemple de test...")
        return []

# Încarcă textele de phishing
phishing_texts = load_phishing_texts()

if not phishing_texts:
    print("Nu s-au putut încărca textele de phishing. Se vor folosi exemple extinse.")
    phishing_texts = [
        "URGENT: Your account will be suspended! Click here to verify: www.fake-bank.com",
        "Congratulations! You've won $1,000,000. Contact us immediately at winner@scam.com",
        "Action required: Update your PayPal information by calling 555-FAKE-NUM",
        "Dear Customer, Your Netflix account expires today. Renew at: https://netflix-renewal.fake",
        "ALERT: Unusual activity detected on your Amazon account. Verify now: amazon-security.scam",
        "You have received a payment of $500. Click to claim: paypal-claim.fake",
        "WARNING: Your Microsoft account has been compromised. Secure it now at: microsoft-help.scam",
        "Congratulations John Smith! You won an iPhone 15 Pro. Call 1-800-SCAM-NOW to claim",
        "Your Chase Bank card is blocked. Unblock it immediately: chase-unblock.fake",
        "FINAL NOTICE: Your subscription to Adobe expires in 24 hours. Renew: adobe-renew.scam"
    ]
    print(f"Se folosesc {len(phishing_texts)} exemple extinse pentru testare.")

# Creează extractorul
extractor = PhishingTextNER(nlp)

print(f"\n{'='*70}")
print("PROCESARE TEXTE DE PHISHING")
print(f"{'='*70}")

# Procesează textele
processed_results = []
global_stats = defaultdict(int)
total_texts = len(phishing_texts)

print(f"\nProcesare {total_texts} texte de phishing...")
print("=" * 60)

for i, phishing_text in enumerate(phishing_texts, 1):
    # Progress indicator
    if i % 100 == 0 or i <= 5:
        progress = (i / total_texts) * 100
        print(f"\n--- MESAJ PHISHING {i}/{total_texts} ({progress:.1f}%) ---")
        print(f"Text: {str(phishing_text)[:100]}...")
    elif i % 50 == 0:
        progress = (i / total_texts) * 100
        print(f"Procesare... {i}/{total_texts} ({progress:.1f}%)")
    
    # Procesează textul
    result = extractor.process_text_only(str(phishing_text))
    
    # Actualizează statisticile globale
    for entity in result['entities']:
        global_stats[entity['label']] += 1
    
    # Afișează rezultatele doar pentru primele 5 și fiecare al 100-lea
    if i <= 5 or i % 100 == 0:
        print(f"Entități spaCy găsite: {len(result['spacy_entities'])}")
        print(f"Entități URGENCY găsite: {len(result['urgency_entities'])}")
        print(f"Total entități: {len(result['entities'])}")
        
        if result['spacy_entities']:
            print("  Entități spaCy:")
            for entity in result['spacy_entities']:
                print(f"    • '{entity['text']}' [{entity['label']}] la pozițiile {entity['start']}-{entity['end']}")
        
        if result['urgency_entities']:
            print("  Entități URGENCY:")
            for entity in result['urgency_entities']:
                print(f"    • '{entity['text']}' [{entity['label']}] la pozițiile {entity['start']}-{entity['end']}")
        
        # Afișează rezultatul formatat
        print(f"\nRezultat formatat:")
        print(f"  {result['formatted_result']}")
    
    processed_results.append({
        'message_id': i,
        'original_text': phishing_text,
        'processed': result
    })

# Salvează rezultatele
def save_phishing_results(results, filename="phishing_ner_results.txt"):
    """Salvează rezultatele pentru textele de phishing"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("REZULTATE EXTRAGERE NER PENTRU MESAJE PHISHING\n")
            f.write("=" * 60 + "\n\n")
            
            for result in results:
                f.write(f"--- MESAJ PHISHING {result['message_id']} ---\n")
                f.write(f"Text original: {result['original_text']}\n")
                f.write(f"Rezultat formatat: {result['processed']['formatted_result']}\n")
                f.write(f"Entități găsite: {len(result['processed']['entities'])}\n")
                
                if result['processed']['entities']:
                    f.write("Detalii entități:\n")
                    for entity in result['processed']['entities']:
                        f.write(f"  - '{entity['text']}' [{entity['label']}] la {entity['start']}-{entity['end']}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        # Salvează în formatul training_data.py
        with open("training_data.py", 'w', encoding='utf-8') as f:
            f.write("training_data = [\n")
            
            for i, result in enumerate(results):
                formatted_result = result['processed']['formatted_result']
                text, entities_dict = formatted_result
                
                # Escape quotes and special characters in text
                escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                
                f.write(f'    ("{escaped_text}", {entities_dict}),\n')
            
            f.write("]\n")
        
        # Salvează și în formatul exact cerut
        with open("formatted_phishing_results.txt", 'w', encoding='utf-8') as f:
            f.write("# Rezultate NER Formatate pentru Mesaje Phishing\n")
            f.write("# Fiecare linie conține: (text, {\"entities\": [(start, end, label), ...]})\n\n")
            
            for result in results:
                f.write(f"{result['processed']['formatted_result']}\n")
        
        print(f"✓ Rezultate salvate în {filename}")
        print(f"✓ Training data salvate în training_data.py (format Python pentru antrenament)")
        print(f"✓ Rezultate formatate salvate în formatted_phishing_results.txt")
        print(f"✓ Total {len(results)} exemple procesate și salvate în format Python")
        return True
        
    except Exception as e:
        print(f"✗ Eroare la salvarea rezultatelor: {e}")
        return False

# Salvează rezultatele
save_phishing_results(processed_results)

# Statistici finale
print(f"\n{'='*70}")
print("STATISTICI FINALE")
print(f"{'='*70}")

print(f"Model folosit: {model_name}")
print(f"Total mesaje phishing procesate: {len(phishing_texts)}")
print(f"Total entități găsite: {sum(global_stats.values())}")
print(f"Tipuri unice de entități: {len(global_stats)}")

if global_stats:
    print(f"\nDistribuția tipurilor de entități:")
    sorted_stats = sorted(global_stats.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_stats:
        description = spacy.explain(label) if spacy.explain(label) else "Necunoscut"
        print(f"  {label:12} {count:3d}  ({description})")
else:
    print("\nNu s-au găsit entități în textele procesate.")

print(f"\n{'='*70}")
print("PROCESARE COMPLETĂ!")
print(f"{'='*70}")