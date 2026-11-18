# -*- coding: utf-8 -*-
"""
detect.py
=========

Roda a detecção (shared/visual.py) para um período específico — por padrão,
sem figuras e sem apagar nenhum .npz.

Uso (a partir da raiz do projeto)::

    python detect.py -y 2026 -m 11 -d 17
    python detect.py -y 2026 -m 11 -d 17 18 -w 8 --server
    python detect.py -y 2026 -m 11 -d 17 --plot      # também gera figuras

IMPORTANTE: --server deve ser igual ao usado no call_download.py, senão a
detecção lê outra pasta de .npz.
"""

import argparse

from shared.visual import main


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Detecção GOES para um período (ano/mês/dia).")
    p.add_argument("-y", "--year", type=int, required=True, help="Ano (ex.: 2026)")
    p.add_argument("-m", "--months", nargs="+", type=int, required=True,
                   help="Mês(es) (ex.: 11)")
    p.add_argument("-d", "--days", nargs="+", type=int, default=None,
                   help="Dia(s) do mês (ex.: 17). Padrão: mês inteiro")
    p.add_argument("-w", "--workers", type=int, default=16,
                   help="Processos em paralelo (padrão: 16)")
    p.add_argument("--server", action="store_true",
                   help="Layout de pastas do servidor (igual ao download)")
    p.add_argument("--plot", action="store_true",
                   help="Gera as figuras (padrão: desligado)")
    p.add_argument("-o", "--output_key", type=str, default="",
                   help="Sufixo das pastas figures/reports")
    a = p.parse_args()

    main(
        server_ON=a.server,
        parallel_ON=True,
        mpi_ON=False,
        n_workers=a.workers,
        output_key=a.output_key,
        plot_ON=a.plot,
        delete_noCloud_file=False,   # nunca apaga .npz
        year=a.year,
        months=a.months,
        days=a.days,
    )