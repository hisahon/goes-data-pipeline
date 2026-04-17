# GOES Data Pipeline (CPTEC/INPE)

Este repositório contém um pipeline robusto e automatizado para o download e processamento de dados dos satélites **GOES-16** e **GOES-19**. O sistema foi projetado para facilitar a pesquisa meteorológica e ionosférica, focando na identificação de sistemas convectivos (MCS) através do Canal 13 (Infravermelho).

---

## 🌟 Destaques do Projeto

* **Inteligência de Satélite**: Seleção automática entre GOES-16 (dados históricos até 03/2025) e GOES-19 (dados operacionais a partir de 04/2025).
* **Alta Performance**: Download e conversão simultâneos utilizando múltiplos núcleos de processamento (`ProcessPoolExecutor`).
* **Pipeline End-to-End**: O script realiza o download do NetCDF, aplica o recorte espacial (crop), converte para NumPy e deleta o arquivo bruto automaticamente para economizar espaço em disco.
* **Compatibilidade Multiplataforma**: Caminhos de diretório inteligentes que se adaptam entre Windows e Linux (Servidor/Local).

[Image of satellite data processing workflow from raw NetCDF to processed NumPy arrays]

---

## 📂 Arquitetura do Repositório

| Arquivo | Função Principal |
| :--- | :--- |
| `download.py` | Interface de linha de comando e lógica de download paralelo. |
| `detector.py` | Leitura de arquivos NetCDF e detecção de clusters (DBSCAN). |
| `converter.py` | Conversão e compressão de arrays para formato `.npz`. |
| `.gitignore` | Bloqueia o upload de arquivos binários pesados e temporários. |

---

## 🛠️ Requisitos de Sistema

### Bibliotecas Python
```bash
pip install numpy requests beautifulsoup4 tqdm xarray scipy scikit-image matplotlib cartopy
