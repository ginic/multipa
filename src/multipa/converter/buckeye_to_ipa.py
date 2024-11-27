"""Convert utterances from the Buckeye corpus transcription alphabet to IPA"""

from phonecodes import phonecodes

BUCKEYE_INTERRUPT_SYMBOL = "U"


def buckeye_to_ipa(buckeye_transcription: str, is_keep_interrupts: bool = False):
    """Returns the IPA version of the Buckeye transcription, accounting for interrupts symbolized
    with an upper case 'U'.
    In the original corpus this includes annotations like EXCLUDE, VOCNOISE and EXT.

    Args:
        buckeye_transcription (str): whitespace delimited string of characters in the Buckeye transcription alphabet
        is_keep_interrupts (bool): Set to True to keep the interrupt symbol in IPA output. Defaults to False

    >>> buckeye_to_ipa("U ah m")
    'U ʌ m'
    """
    buckeye_sym_list = buckeye_transcription.split()
    final_segments = []
    start_index = 0
    # Check for interrupts and convert substrings
    for i, sym in enumerate(buckeye_sym_list):
        if sym in [BUCKEYE_INTERRUPT_SYMBOL, "VOCNOISE", "UNKNOWN"]:
            # Convert string prior to interruption
            if i != start_index:
                ipa_seg = phonecodes.buckeye2ipa(" ".join(buckeye_sym_list[start_index:i]))
                final_segments.append(ipa_seg)
            if is_keep_interrupts:
                final_segments.append(BUCKEYE_INTERRUPT_SYMBOL)
            start_index = i + 1
        # Handle last symbols in the string
        elif i == len(buckeye_sym_list) - 1:
            ipa_seg = phonecodes.buckeye2ipa(" ".join(buckeye_sym_list[start_index:]))
            final_segments.append(ipa_seg)

    if len(final_segments) > 0:
        return " ".join(final_segments)
    else:
        return ""
