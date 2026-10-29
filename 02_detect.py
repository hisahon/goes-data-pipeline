# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:43:00 2025

@author: hisas
"""

#Gera todas as figuras faltantes, apenas de run

from shared.visual import main
import argparse


if __name__ == "__main__":

    
    main(
        server_ON=False,
        parallel_ON=True,
        mpi_ON=False,
        n_workers = 16,
        output_key="",
    )
    
    # parser = argparse.ArgumentParser(description="Processa dados GOES com opções de ambiente e paralelismo.")
    # parser.add_argument("--server", action="store_true")
    # parser.add_argument("--parallel", action="store_true")
    # parser.add_argument("--mpi", action="store_true")
    # parser.add_argument("-o", "--output_key", type=str, default="")
    # parser.add_argument("-w", "--workers", type=int, default=None,
    #                     help="Parallel workers (will be capped to CPU-1; default: auto)")
    # args = parser.parse_args()
    
    # main(
    #     server_ON=args.server,
    #     parallel_ON=args.parallel,
    #     mpi_ON=args.mpi,
    #     n_workers = args.workers,
    #     output_key=args.output_key,
    # )
    
# example of usage: 
    # source /mpi-env/bin/activate
    # python ~/cptec_codes/visual_terminal.py --parallel --server -w 40 -o big