# GOES Data Pipeline

Este repositório contém um pipeline automatizado em Python para o download, conversão e processamento de dados dos satélites **GOES-16** e **GOES-19**. O foco do projeto é a identificação de aglomerados convectivos utilizando dados infravermelhos (Canal 13).

## 🚀 Funcionalidades

* **Seleção Automática de Satélite**: O sistema alterna automaticamente entre GOES-16 e GOES-19 com base na data solicitada (transição em Abril de 2025).
* **Download Multiprocessado**: Utiliza `ProcessPoolExecutor` para realizar downloads paralelos e otimizar o tempo de transferência.
* **Conversão para NumPy**: Transforma arquivos NetCDF brutos em arquivos `.npz` compactados e otimizados para leitura rápida.
* **Detecção de Clusters**: Implementa algoritmos de segmentação e detecção de aglomerados frios (DBSCAN/RegionProps).
* **Recorte Espacial (Crop)**: Realiza o recorte automático dos dados para a região da América do Sul utilizando máscaras pré-definidas.

## 📂 Estrutura do Projeto

* `download.py`: Script principal para interface de linha de comando (CLI) e gerenciamento de tarefas de download.
* `detector.py`: Módulo responsável pela leitura de arquivos NetCDF e detecção de sistemas convectivos.
* `converter.py`: Utilitário para conversão e quantização de dados para o formato `.npz`.
* `.gitignore`: Configuração para impedir o envio de dados pesados (`.nc`, `.npz`) para o GitHub.

## 🛠️ Instalação

Certifique-se de ter o Python instalado e as seguintes dependências:

```bash
pip install numpy requests beautifulsoup4 tqdm xarray scipy scikit-image matplotlib cartopy
