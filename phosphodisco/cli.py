import argparse
import logging
import phosphodisco

fmt = "%(asctime)s:%(levelname)s:%(message)s"
logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")
logging.captureWarnings(True)

def _set_up_logger(path):
    logger = logging.getLogger("cli")
    fh = logging.FileHandler("%s.log" % path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    fmter = logging.Formatter(fmt)
    fh.setFormatter(fmter)
    ch.setFormatter(fmter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _make_parser():
    parser = argparse.ArgumentParser(prog="blacksheep", description="")
    parser.add_argument("--version", "-v", action="version", version="%(prog)s 0.0.1")

    subparsers = parser.add_subparsers(dest="which")
    subparsers.required = True

    normalize = subparsers.add_parser(
        "normalize",
        description="Takes an unnormalized values table and uses median of ratios normalization "
                    "to normalize. Saves a log2 normalized table appropriate for BlackSheep "
                    "analysis."
    )
    normalize.add_argument(
        "unnormed_values",
        type=_is_valid_file,
        help="Table of values to be normalized. Sites/genes as rows, samples as columns. "
    )


def _main(args: Optional[List[str]] = None):
    if args is None:
        args = sys.argv[1:]
    args = _make_parser().parse_args(args)

    logger = _set_up_logger(args.output_prefix)

    logger.info("Running deva in %s mode" % args.which)
    for arg in vars(args):
        logger.info("Parameter %s: %s" % (arg, getattr(args, arg)))

    if args.which == "outliers_table":
        df = parsers.read_in_values(args.values)
        make_outliers_table(
            df,
            iqrs=args.iqrs,
            up_or_down=args.up_or_down,
            aggregate=~args.do_not_aggregate,
            save_outlier_table=True,
            save_frac_table=args.write_frac_table,
            output_prefix=args.output_prefix,
            ind_sep=args.ind_sep,
        )


if __name__ == "__main__":
    _main()

