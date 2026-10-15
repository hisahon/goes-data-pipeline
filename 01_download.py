
# -*- coding: utf-8 -*-
"""
call_download.py
================

Lançador do estágio de download a partir da raiz do projeto.

O ``download.py`` vive em ``shared/`` (camada de biblioteca/execução). Este
wrapper fino apenas coloca ``shared/`` no ``sys.path`` e delega para o
``argparse`` que já existe em ``download.main_terminal()`` — todas as flags
(``-y``, ``-m``, ``-d``, ``-c``, ``-w``, ``-t``, ``--server``) são repassadas
diretamente, sem duplicar a definição da CLI.

Uso (a partir da raiz do projeto)::

    python call_download.py -y 2021 -m 10 -d 23 -w 8
    python call_download.py -y 2025 -m 3 4 5 6 7 8 9 10 --server -w 10

IMPORTANTE
----------
O ``download.py`` carrega ``mask_SouthAmerica.npz`` com caminho **relativo**
(``np.load("mask_SouthAmerica.npz")``). Portanto execute este script a partir
do diretório que contém o ``mask_SouthAmerica.npz`` (normalmente a própria
raiz do projeto). No Linux os diretórios de saída são ancorados em ``~`` e não
dependem de onde você executa; no Windows eles são relativos ao diretório
atual.
"""

from pathlib import Path

# ``shared/`` fica ao lado deste arquivo — resolvido a partir da localização
# do script, independente do diretório de onde o ``python`` foi chamado.
ROOT = Path(__file__).resolve().parent


# Import só é possível depois de ajustar o sys.path acima.
from shared.download import main_terminal, main_manual  # noqa: E402


if __name__ == "__main__":
    main_terminal()
    # main_manual()