from typing import Optional, List, Iterable
import sys
import logging
import argparse
import yaml
import oyaml
import numpy as np
import phosphodisco
import pathlib
import importlib
from io import BytesIO
import pkgutil
import subprocess

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('PhosphoDisco')


def _make_parser(fun=None, help_text=None):
    """
    Makes a parser for the clis.
    """
    parser = argparse.ArgumentParser(prog="phosphodisco", description=help_text)
    parser.add_argument(
        "--version", "-v", action="version", version="%s" % phosphodisco.__version__
    )
    if fun == 'generate_config':
        parser.add_argument(
                "--config_path", default='phdc_custom_config.yml', type=pathlib.Path, 
                help='path where config should be output'
        )        
        parser.add_argument(
                "--template", default=0, choices=[0, 1, 2], type=int,
                help='which config template to use - 0 was used for phosphodisco publication, 1 contains additional clustering methods (LouvainCluster and LeidenCluster), 2 is the template used for the demo/tutorial'
        )        
        parser.add_argument(
                "--phospho", type=pathlib.Path, required=True, 
                help='path to phospho file'
        )
        parser.add_argument(
                "--protein", type=pathlib.Path, required=True, 
                help='path to protein file'
        )
        parser.add_argument(
                "--sample_table", type=pathlib.Path, default=None,
                help='path to a sample annotation table csv file. First column has sample ids.'
        )
        parser.add_argument(
                "--sample_columns", type=list, default=None,
                help='which columns from sample annotation table to use. Will use all cols if none provided.'
        )
        parser.add_argument(
                "--min_common_values", type=int, default=6, help=''
        )
        parser.add_argument(
                 "--top_stdev_quantile", type=float, default=0.5, help=''
        )
        parser.add_argument(
                "--na_frac_threshold", type=float, default=0.25, help=''
        )

    elif fun == 'run':
        parser.add_argument(
                "--config_file", type=pathlib.Path, required=True,
                help='path to snakemake config file to be used by pipeline. Use phdc_generate_config to geneerate one from a template'
        )
        parser.add_argument(
                "--cores", type=int, default=3, 
                help='how many cores to use. default=3'
        )
        parser.add_argument(
                "--jobs", type=int, default=20, 
                help='how many jobs to run at a time. default=20'
        )
        parser.add_argument(
                "--cluster_config", type=pathlib.Path, 
                help='snakemake cluster config file'
        )
        parser.add_argument(
                "--dry_run", action='store_const', const='-n', default='',
                help='perform a dry-run of the pipeline, i.e. just print what would be run'
        )
        parser.add_argument(
                "--cluster_submit_command", default='',
                help='cluster submit command to feed to snakemake. Will run in local mode if not provided.\nslurm example:\n"sbatch --mem={cluster.mem} -t {cluster.time} -o {cluster.output} --cpus-per-task {cluster.cpus-per-task}"'
        )
        

# snakemake --snakefile phdc.smk --cores 3 -n --configfile config-custom.yml --cluster-config cluster.json

    return parser

def run():
    """
    Runs the snakemake phosphodisco workflow from the command line.
    """
    help_text="""Runs the snakemake phosphodisco workflow from the command line."""
    logger.info('run()\n'+ help_text)
    parser = _make_parser(fun='run', help_text=help_text)
    args = parser.parse_args()
    phdc_path = importlib.util.find_spec('phosphodisco')
    phdc_path = pathlib.Path(phdc_path.origin).parent
    snakefile = phdc_path / 'phdc.smk'
    snakemake_cmd = [
        'snakemake', 
        '--snakefile', snakefile, 
        '--cores', str(args.cores), 
        '--configfile', args.config_file,
        args.dry_run,
        f'--cluster-config {args.cluster_config}' if args.cluster_config else '',     
        f"--cluster '{args.cluster_submit_command}'" if args.cluster_submit_command else '',
        f"-j {args.jobs}" if args.jobs else '',

    ]
    snakemake_cmd = list(filter(lambda it: it!='', snakemake_cmd))
    logger.info(' '.join(['command run:\n'] + [str(i) for i in snakemake_cmd]))
    subprocess.run(
        ' '.join([str(i) for i in snakemake_cmd]),
        shell=True
    )

def generate_config():
    """
    Makes a copy of the config template and modifies it according to the flags.
    """
    help_text="""Generates a config file to be used by phdc_run."""
    parser = _make_parser(fun='generate_config', help_text=help_text)
    args = parser.parse_args()
    config_template_dict = {
            0:'data/config-custom_template0.yml', 
            1:'data/config-custom_template1.yml',
            2:'data/config-custom_template2.yml'
            }
    config_template = BytesIO(pkgutil.get_data('phosphodisco', config_template_dict[args.template])) 
    template_yml = oyaml.load(config_template, Loader=oyaml.FullLoader)
    if args.template == 2:
        # template 2 is used for the demo
        # only the paths have to be set, everything else is set in the template already
        template_yml['input_phospho'] = pathlib.Path(__file__).parent / 'data' / 'demo' / 'BRCA_v5.2__selected_phospho.csv'
        template_yml['input_protein'] = pathlib.Path(__file__).parent / 'data' / 'demo' / 'BRCA_v5.2__selected_protein.csv'
    else:
        template_yml['input_phospho'] = str(args.phospho)
        template_yml['input_protein'] = str(args.protein)
        template_yml['std_quantile_threshold'] = args.top_stdev_quantile
        template_yml['min_common_vals'] = args.min_common_values
        template_yml['na_frac_threshold'] = args.na_frac_threshold
       #add sample table and sample columns info 
        template_yml['sample_annotations_csv'] = args.sample_table
        template_yml['sample_annot_cols_for_normalization'] = args.sample_columns

    # warnings in case user mistypes phospho/protein paths:
    for field, path in {'phospho':args.phospho, 'protein':args.protein}.items():
        if not pathlib.Path(path).exists():
            logger.warning(f'The following {field} path does not exist: {path}')
    # write new custom config to file
    with open(args.config_path, 'w') as fh:
        fh.write(oyaml.dump(template_yml))


def _main(args: Optional[List[str]] = None):
    if args is None:
        args = sys.argv[1:]
    args = _make_parser().parse_args(args)
    logger.info("Running phosphodisco")

    for arg in vars(args):
        logger.info("Parameter %s: %s" % (arg, getattr(args, arg)))

    output_prefix = args.output_prefix
    phospho = phosphodisco.parsers.read_phospho(args.phospho)
    protein = phosphodisco.parsers.read_protein(args.protein)
    min_common_values = args.min_common_values

    if args.normed_phospho:
        normed_phospho = phosphodisco.parsers.read_phospho(args.normed_phospho)
    else:
        normed_phospho = None

    if args.modules:
        modules = phosphodisco.parsers.read_phospho(args.modules)
    else:
        modules = None

    if args.additional_kwargs_yml:
        with open(args.additional_kwargs_yml, 'r') as fh:
            additional_kwargs_yml = yaml.load(fh, Loader=yaml.FullLoader)
    else:
        additional_kwargs_yml = {}

    data = phosphodisco.ProteomicsData(
        phospho=phospho,
        protein=protein,
        min_common_values=min_common_values,
        normed_phospho=normed_phospho,
        modules=modules,
    )
    logger.info("Instantiated ProteomicsData")
    if args.normed_phospho is None:
        logger.info('Normalizing phospho by protein')
        data.normalize_phospho_by_protein(
            **additional_kwargs_yml.get('normalize_phospho_by_protein', {})
        )
        logger.info("Finished normalizing phospho by protein")
        #TODO make it so this doesn't keep writing this file
        data.normed_phospho.to_csv('%s.normed_phospho.csv' % output_prefix)
    if data.normed_phospho.isnull().any().any():
        logger.info("Imputing missing values")
        data.impute_missing_values(**additional_kwargs_yml.get('impute_missing_values', {}))
        logger.info("Finished imputing missing values")
        data.normed_phospho.to_csv('%s.normed_phospho.csv' % output_prefix)
    if args.top_stdev_percent < 100:
        data.normed_phospho = data.normed_phospho.loc[
            data.normed_phospho.std(axis=1)<np.percentile(
                data.normed_phospho.std(axis=1), 100-args.top_stdev_percent
            )
        ]
        data.normed_phospho.to_csv('%s.normed_phospho.top%sstdev_percent.csv' % (output_prefix, args.top_stdev_percent))
    
    
    if args.stop_before_modules:
        logger.info('Stopping before calculating modules, PhosphoDisco done')
        return None
    
    if args.modules is None:
        logger.info("Assigning modules")
        data.assign_modules(
            force_choice=True, **additional_kwargs_yml.get('assign_modules', {})
        )
        data.modules.to_csv('%s.modules.csv' % output_prefix)
        logger.info("Finished assigning modules")
    data.calculate_module_scores(
        **additional_kwargs_yml.get('calculate_module_scores', {})
    )
    logger.info("Calculated module scores")

    if args.putative_regulator_list:
        with open(args.putative_regulator_list, 'r') as fh:
            putative_regulator_list = [gene.strip() for gene in fh.readlines()]
        data.collect_possible_regulators(
            putative_regulator_list, **additional_kwargs_yml.get('collect_possible_regulators', {})
        )
        logger.info("Collected possible regulators")
        data.calculate_regulator_association(
            **additional_kwargs_yml.get('calculate_regulator_coefficients', {})
        )
        data.regulator_coefficients.to_csv('%s.putative_regulator_coefficients.csv' % output_prefix)
        logger.info("Calculated regulator coefficients")

    if args.annotations:
        if args.annotation_column_types is None:
            logger.error(
                'Annotations were provided, but no column labels were provided. Cannot continue '
                'with annotation association calculations. '
            )
            return None
        annotations = phosphodisco.parsers.read_annotation(args.annotations)
        with open(args.annotation_column_types, 'r') as fh:
            annotation_column_types = [col.strip() for col in fh.readlines()]
        data.add_annotations(
            annotations=annotations,
            column_types=annotation_column_types
        )
        logger.info("Added annotations")
        data.calculate_annotation_association(
            **additional_kwargs_yml.get('calculate_annotation_association', {})
        )
        data.annotation_association.to_csv('%s.annotation_association.csv' % output_prefix)
        logger.info("Calculated annotation association. ")
        
    logger.info("PhosphoDisco done")


if __name__ == "__main__":
    _main()
