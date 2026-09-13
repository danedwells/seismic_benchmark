EDT_SIGMA_S    = config.BENCHMARK_PARAMS['edt_sigma_s']
SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']



DTT_WEIGHT     = config.BENCHMARK_PARAMS['dtt_weight']
EDT_TAG        = f'edt_{EDT_SIGMA_S}'
S_TAG          = f'sig_{SIGMA_S}'

_VARY_EDT      = os.environ.get('VARY_EDT', '0') == '1'
_VARY_SIG      = os.environ.get('VARY_SIG', '0') == '1'

if _VARY_EDT == True & _VARY_SIG == True:
    raise Exception("Cannot vary both EDT and Sigma at the same time")
elif _VARY_EDT == True:
    OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output',  'time_independent', EDT_TAG, f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'time_independent', EDT_TAG, f'max_trigs_{MAX_TRIGS}')
elif _VARY_SIG == True:
    OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output',  'time_independent', S_TAG, f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'time_independent', S_TAG, f'max_trigs_{MAX_TRIGS}')
