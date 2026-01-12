def vprint(verbose: bool, message: str) -> None:
    """Prints message if verbose mode is enabled."""
    if verbose:
        print(message)
