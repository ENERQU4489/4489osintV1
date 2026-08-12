# 4489 OSINT Tool v1 — Silnik Geolokalizacji AI

> **Monochromatyczne Narzędzie OSINT & Geolokalizacji Obrazów ze wsparciem GUI, CLI oraz Bazy Społeczności**

**4489 OSINT Tool v1** pozwala na precyzyjne ustalanie współrzędnych GPS zdjęć ulicznych z dokładnością do kilku metrów. Narzędzie wykorzystuje dwuetapowy potok: **MegaLoc** (CVPR 2025) / **MixVPR** do wyszukiwania globalnego oraz **MASt3R** (ECCV 2024) do gęstego dopasowania geometrii 3D.

---

## 🇵🇱 Główne Funkcje

1. **Wsparcie dla języka polskiego**: Pełne tłumaczenie interfejsu GUI (etykiety, przyciski, menu kontekstowe, poradnik, pomoc) oraz wiersza poleceń CLI.
2. **Monochromatyczny Interfejs High-Contrast**: Głęboka czerń (`#000000`) i czysta biel (`#ffffff`) zapewniające najwyższą czytelność i nowoczesny wygląd OSINT.
3. **Paski Ładowania & Wskaźniki Procentowe**: Wizualny pasek postępu z dokładną wartością procentową (`0%` - `100%`) w GUI i CLI.
4. **Pełna Obsługa CLI**: Możliwość prowadzenia przeszukiwania, budowania baz i porównywania zdjęć z poziomu wiersza poleceń.
5. **Format `.4489` i Społeczność (Hub)**: Udostępnianie i pobieranie gotowych baz miast w formacie `.4489` (oraz kompatybilnych `.noname` / `.netryx`).

---

## 🚀 Szybki Start

### Interfejs Graficzny (GUI)
```bash
# Windows
run.bat

# Dedykowany plik startowy
python osint4489.py
```

---

## 💻 Interfejs Wiersza Poleceń (CLI)

### 1. Wyszukiwanie lokalizacji zdjęcia
```bash
python osint4489.py search --image zdjecie.jpg --lat 40.7132 --lon -74.0025 --radius 5.0
```
Opcje dodatkowe:
- `--encoder megaloc|mixvpr` (domyślnie: `megaloc`)
- `--top-k 10`
- `--no-mast3r` (pomiń dopasowanie 3D)
- `--output wyniki.json` (zapisz wyniki do pliku JSON)

### 2. Budowanie nowej bazy obszaru
```bash
python osint4489.py index --lat 40.7132 --lon -74.0025 --radius 2.0 --res 300 --encoder megaloc
```

### 3. Wyświetlenie listy lokalnych baz
```bash
python osint4489.py list-indexes
```

### 4. Porównanie dwóch zdjęć (MASt3R 3D)
```bash
python osint4489.py match --image1 zdjecie1.jpg --image2 zdjecie2.jpg
```

### 5. Obsługa Bazy Społeczności (Hub)
```bash
# Wyświetlenie baz w chmurze
python osint4489.py hub list

# Pobranie bazy miasta
python osint4489.py hub download --id moscow
```

---

## 🔧 Instalacja

### Windows
1. Uruchom `setup.bat` w celu automatycznego przygotowania środowiska virtualenv i bibliotek PyTorch + MASt3R.
2. Uruchom `run.bat` lub `python osint4489.py`.

### Linux / macOS
```bash
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
python osint4489.py
```

---

## 📜 Licencja
Licencja MIT. Udostępniane bazy danych na licencji CC-BY-4.0.
