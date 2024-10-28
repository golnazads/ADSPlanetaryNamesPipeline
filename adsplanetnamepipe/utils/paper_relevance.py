import regex

from adsplanetnamepipe.utils.common import EntityArgs, Synonyms


class PaperRelevance():
    """
    a class that calculates the relevance score of a paper based on various criteria

    this class uses regular expressions to match target and feature type terms,
    and considers factors such as the paper's database, journal, and the presence
    of relevant terms to compute a relevance score
    """

    def __init__(self, args: EntityArgs):
        """
        initialize the PaperRelevance class

        :param args: configuration arguments containing target and feature type information
        """
        self.synonyms = Synonyms()
        self.re_match_target = regex.compile(r'\b(%s)\b' % self.synonyms.get_target_terms(args.target), flags=regex.IGNORECASE)
        self.re_match_feature_type = regex.compile(r'\b(%s)\b' % self.synonyms.get_feature_type_terms([args.feature_type, args.feature_type_plural]), flags=regex.IGNORECASE)

    def forward(self, text: str, bibstem: str, databases: str, astronomy_main_journals: list, len_existing_wikidata: int) -> float:
        """
        calculate the relevance score of a paper

        :param text: the text content of the paper
        :param bibstem: the bibliographic stem of the paper
        :param databases: string containing the ads collection the paper is in
        :param astronomy_main_journals: list of main astronomy journals
        :param len_existing_wikidata: number of existing Wikidata terms in the paper
        :return: float representing the calculated relevance score of the paper
        """
        # threshold for num of targets, feature type, and wikidata terms
        threshold = text.count(' ') * 0.001

        # get number of times target and feature type appeared in the text,
        # each is worth 0.2 in scoring if above threshold
        target_terms_len = len(self.re_match_target.findall(text))
        feature_type_terms_len = len(self.re_match_feature_type.findall(text))

        in_astronomy_main_journals = any(journal == bibstem for journal in astronomy_main_journals)

        # the threshold and weights have been determined empirically, by analyzing several thousands scores
        return int('astronomy' in databases) * 0.2 + \
               int(in_astronomy_main_journals) * 0.2 + \
               int(target_terms_len > threshold) * 0.2 + \
               int(feature_type_terms_len > 0) * 0.2 + \
               int(len_existing_wikidata > threshold) * 0.2

