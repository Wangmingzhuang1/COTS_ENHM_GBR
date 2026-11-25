"""Reproduce the dated-survey analysis and optional sensitivity/figure stages."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true',help='Audit results already present in results/tables')
    parser.add_argument('--robustness',action='store_true',help='Run Raw/ENHM/OLS, lag and phase sensitivity checks')
    parser.add_argument('--figures','--figs',action='store_true',help='Build all current paper figures; requires external maps and spatial inputs')
    parser.add_argument('--environment',action='store_true',help='Export exact environmental source data; requires external environmental inputs')
    parser.add_argument('--supplementary',action='store_true',help='Run additional time-domain diagnostics; these are not the primary spectral evidence')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'scripts'))
    from prepare_ecology import prepare
    prepare()
    stages=[] if args.audit_only else ['01_bsra_analysis.py']
    stages.append('12_consistency_audit.py')
    if args.robustness or args.figures:
        stages.append('13_robustness_null_comparison.py')
    if args.environment:
        stages.append('17_environment_source_data.py')
    if args.figures:
        stages += ['04_cots_overview.py','02_paper_figures.py','04_spatial_heterogeneity.py','05_propagation.py','06_mechanism_fig6.py','16_conceptual_susceptibility_schematic.py']
    for stage in stages:
        print('Running '+stage,flush=True)
        subprocess.run([sys.executable,'-B',str(root/'scripts'/stage)],cwd=root,check=True)
    if args.supplementary:
        for module in ['05_reef_level_exclusivity','06_ipo_outbreak_transitions']:
            subprocess.run([sys.executable,'-B','-m','pnas_aug.scripts.'+module],cwd=root,check=True)

if __name__=='__main__':
    main()
