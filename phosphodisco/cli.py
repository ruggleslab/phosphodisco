from typing import Optional, List, Iterable
import sys
import logging
import argparse
import yaml
import phosphodisco

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('PhosphoDisco')

def _make_parser():
    parser = argparse.ArgumentParser(prog="phosphodisco", description="")
    parser.add_argument(
        "--version", "-v", action="version", version="%s" % phosphodisco.__version__
    )
    parser.add_argument(
        "phospho", type=str, help=''
    )
    parser.add_argument(
        "protein", type=str, help=''
    )
    parser.add_argument(
        "--output_prefix", type=str, default='phdc', help=''
    )
    parser.add_argument(
        "--min_common_values", help=''
    )
    parser.add_argument(
        "--normed_phospho", type=str, help=''
    )
    parser.add_argument(
        "--modules", type=str, help=''
    )
    parser.add_argument(
        "--stop_before_modules", action='store_true', help=''
    )
    parser.add_argument(
        "--putative_regulator_list", type=str, help=''
    )
    parser.add_argument(
        "--annotations", type=str, help=''
    )
    parser.add_argument(
        "--annotation_column_types", type=str, help=''
    )
    parser.add_argument(
        "--additional_kwargs_yml", type=str, help=''
    )
    return parser


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
        data.normed_phospho.to_csv('%s.normed_phospho.csv' % output_prefix)
    if data.normed_phospho.isnull().any().any():
        logger.info("Imputing missing values")
        data.impute_missing_values(**additional_kwargs_yml.get('impute_missing_values', {}))
        logger.info("Finished imputing missing values")
        data.normed_phospho.to_csv('%s.normed_phospho.csv' % output_prefix)
    
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
        data.calculate_regulator_coefficients(
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
