# Analysis of Battery Bidding Behaviours Through Classification Algorithms in UK Power Markets

This repository presents a study applying machine learning classification to identify strategic “fingerprints” of battery energy storage systems (BESS) across multiple UK electricity markets during 2024.

## Overview

Battery operators in the UK participate in several markets (ancillary services, balancing mechanism, wholesale) to stack revenues. This research investigates:

1. Can ML classifiers identify operators from daily bidding patterns?
2. What distinctive bidding strategies exist?
3. How do operators stack bids across different markets?

## Data Sources

| Market | Source | Key fields |
|--------|--------|-------------|
| EAC (Enduring Auction Capability) | NESO Data Portal | delivery window, auction unit, quantity, service type |
| Balancing Mechanism | ELEXON BMRS | settlement period, BMU, levelFrom/To |
| Wholesale (proxy) | ELEXON Physical Notification | settlement period, levelFrom/To |
| Balancing Reserve | NESO Sell Orders | delivery window, quantity |
| BMU metadata | ELEXON BMU List | leadPartyName, fuelType, capacity |

**Period:** 2024-01-01 – 2024-12-31  
**Operators:** Tesla, EDF Energy Customers, Statkraft Markets, Limejump Energy, Flexitricity, and others.

## Methodology

### Data Processing
- Raw volumes (MW/h) normalised by BMU generation/demand capacity to focus on **strategy, not asset scale**.
- Data reshaped to cross‑sectional format: each row = one day + BMU, columns = 48 half‑hourly settlement periods with normalised bid volumes.
- Excluded: Capacity Market (no public bidding data), imbalance settlements, STOR (small sample).

### Classification
Two algorithms compared:
- **Naive Bayes** – baseline (assumes feature independence)
- **Random Forest** – main classifier (captures non‑linear interactions)

### Evaluation (handling class imbalance)
- Overall Accuracy
- Weighted F1‑score
- Macro F1‑score (primary metric – treats all operators equally)

## Results

### Classification Performance

Random Forest substantially outperforms Naive Bayes, confirming that bidding strategies rely on complex, interdependent temporal patterns.

| Market | Random Forest | | | Naive Bayes | | |
|--------|---------------|----|----|---------------|----|----|
| | Acc | Macro F1 | Weighted F1 | Acc | Macro F1 | Weighted F1 |
| Dynamic Containment | 0.87 | 0.78 | 0.87 | 0.29 | 0.24 | 0.29 |
| Dynamic Moderation | 0.91 | 0.79 | 0.90 | 0.29 | 0.22 | 0.30 |
| Dynamic Regulation | 0.95 | 0.93 | 0.95 | 0.41 | 0.32 | 0.44 |
| Quick Reserve | 0.94 | 0.87 | 0.94 | 0.85 | 0.72 | 0.86 |
| Balancing Reserve | 0.97 | 0.69 | 0.97 | 0.67 | 0.38 | 0.74 |
| Balancing Mechanism | 0.77 | 0.64 | 0.76 | 0.30 | 0.24 | 0.32 |
| Physical Notification | 0.67 | 0.48 | 0.64 | 0.20 | 0.15 | 0.17 |

**Key takeaway:** High Macro F1 in Dynamic Regulation (0.93) and Quick Reserve (0.87) proves operators have distinct, identifiable bidding fingerprints.

### Example Confusion Matrix (Random Forest VS Naive Bayes – Dynamic Containment)

<img width="1104" height="938" alt="cm" src="https://github.com/user-attachments/assets/4f4d4b6f-4d99-446c-90b1-f63995055fed" />

<img width="1104" height="938" alt="cm nb" src="https://github.com/user-attachments/assets/945969ee-42fd-4d40-b05e-a8ec4d384b9f" />

### Overall Market Participation Profiles

Average normalised volume per settlement period across all operators and markets:
<img width="891" height="744" alt="82797a16-a85f-445f-afb3-ce4bec8e08e0" src="https://github.com/user-attachments/assets/6c28d857-890c-4437-89d6-0c620edaa4ea" />

*Each service plays a distinct role in revenue stacking – e.g., Dynamic Containment charges during morning/afternoon, Physical Notification discharges during evening peak.*

### Bid Volume Distributions

Distributions of normalised bidding volumes across all markets:
<img width="1429" height="1229" alt="download (73)" src="https://github.com/user-attachments/assets/1edeb5f8-93d6-480d-b0fa-a6fdec580c20" />

*Most volumes cluster near zero; Quick Reserve and Balancing Reserve show secondary peaks at fractional capacities.*

### Feature Importance

Feature importance reveals discriminative settlement periods:
- **Quick Reserve & Dynamic Regulation** – extreme concentration on period 48 (end of day)
- **Balancing Mechanism** – early periods 1–3 most important
- **Most services** – peak importance during high‑value periods 17–22 and 35–45

### Decision Path Clustering (Example Market)

Sub‑patterns within a strategic group identified via clustering (k‑means, DBSCAN, spectral clustering evaluated by silhouette score):

<img width="1194" height="737" alt="908ee552-7b9e-4ca7-b2ce-5c5c9c3c9e60" src="https://github.com/user-attachments/assets/27974582-d242-4707-a07e-9d16550a7c4b" />
*Different colours indicate tactical sub‑patterns within the same operator group.*

### Company Sub‑Cluster Example (Flexitricity)

Flexitricity’s assets show high association with Auction Unit (Cramér’s V > 0.87) and low with Month (< 0.53), meaning each battery follows a customised, asset‑specific strategy rather than a uniform portfolio approach.
<img width="1489" height="984" alt="felx plot" src="https://github.com/user-attachments/assets/5c7cd8b0-a353-47b9-989e-f60511a7587c" />

### Continuous Group Table – EDF Energy Customers

EDF assets form consistent bidding groups across markets. Example: Group 1 (BR‑led strategy) and Group 3 (DC/DM‑driven).
<img width="1148" height="840" alt="9e6c963f-4308-4b33-bbde-057f86c1138e" src="https://github.com/user-attachments/assets/23888a76-1c68-4d9f-844a-74c5dc898781" />
<img width="1189" height="690" alt="download (57)" src="https://github.com/user-attachments/assets/a23b38e3-ffd7-4f72-863a-453074400e6a" />

| Group | Assets | Strategy |
|-------|--------|----------|
| Group 1 | AG-PEDF01, NURSB-1 | Strong Balancing Reserve presence; PN evening discharge; DR consistently negative |
| Group 3 | PINFB-2, PINFB-3, PINFB-4 | Positive DC/DM overnight and late day; negative during midday transitions; DR charging |

### Cross‑Market Patterns – Key Operators

Each operator exhibits a distinct cross‑market “fingerprint”.

#### Tesla Motors Limited

<img width="1704" height="846" alt="75b9373f-465a-46e2-a9d8-5521ffa36cd0" src="https://github.com/user-attachments/assets/0b2c025c-4629-48bf-afb8-ae34de1d090d" />

*Balanced portfolio: DC/DM/DR for charging and SOC management; PN for evening peak discharge; BR as support.*

#### Statkraft Markets GmbH

<img width="1495" height="690" alt="download (17)" src="https://github.com/user-attachments/assets/3f7146c9-1626-42ee-850c-eb7499047698" />

*QR/BR‑led discharge; DR serves as charging layer; PN opportunistic around evening peak.*

#### EDF Energy Customers Limited
<img width="1495" height="690" alt="download (55)" src="https://github.com/user-attachments/assets/fff8e3e0-bf5b-4a98-9678-01c65e184823" />

<img width="1494" height="690" alt="download (30)" src="https://github.com/user-attachments/assets/bd2c5669-dc42-415f-b7ba-75380d67a0ec" />

*BR/PN‑led discharge; QR at night; DR provides complementary charging.*

#### Limejump Energy Limited
<img width="1495" height="690" alt="download (7)" src="https://github.com/user-attachments/assets/e330a961-57cc-4b57-ac2e-48ecf5902edd" />

<img width="1495" height="690" alt="download (8)" src="https://github.com/user-attachments/assets/5b55ede8-daf2-41be-92b8-50164aa6035e" />

*High‑volume dual‑directional bidding, often heavy unidirectional charging.*

## Conclusions

- Random Forest successfully identifies operator‑specific strategies (Macro F1 up to 0.93).
- Strategic patterns align with market structures – EFA blocks for DC/DM; 60‑minute energy requirement forces convergence in Dynamic Regulation (Tesla).
- Multiple viable business models coexist: portfolio diversifier (Tesla), concentrated peak discharge (Statkraft/EDF), high‑volume contrasting bids (Limejump), asset‑specific flexibility (Flexitricity).

**Limitations:**
- Only BMU‑registered assets.
- Physical Notification is a wholesale proxy (not actual trades).
- Capacity Market excluded; single‑year analysis.
- Lack of study of seasonal/location/capacity factors.
- Lack of investigation into the introduction of Quick Reserve's influence on the market.


## References

- Elexon BMRS – Balancing Mechanism Reporting Service  
- NESO – EAC and Balancing Reserve data  
- Modo Energy – GB BESS Outlook & methodology  
