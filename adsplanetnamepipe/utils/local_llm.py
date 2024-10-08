import requests

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

from adsplanetnamepipe.utils.common import EntityArgs


class LocalLLM():

    """
    a class that interacts with a local large language model to analyze scientific texts

    this class sends requests to the API endpoint to analyze scientific articles
    and determine the probability that a given term refers to a specific feature on a target celestial body
    """

    def __init__(self, args: EntityArgs):
        """
        initialize the LocalLLM class

        :param args: configuration arguments containing feature name, feature type, and target information
        """
        self.args = args

    def forward(self, title: str, abstract: str, excerpt: str) -> float:
        """
        analyze a scientific article and determine the probability of a term referring to a specific feature

        :param title: the title of the scientific article, is a list
        :param abstract: the abstract of the scientific article
        :param excerpt: the specific excerpt to be analyzed
        :return: float representing the probability (between 0 and 1) that the feature name refers to the specified feature type on the target
        """
        if abstract:
            title = ' '.join(title)
            content = f'Consider the following scientific article:\n\n' \
                      f'Title: {title}\n\n' \
                      f'Abstract: {abstract}\n\n' \
                      f'Task: Please analyze the following excerpt: {excerpt}\n\n' \
                      f'Based on the context provided, answer what is the probablity (just give a value between 0 and 1), with the limited information, '\
                      f'that the term "{self.args.feature_name}" refers to a "{self.args.feature_type}" on the {self.args.target}?\n\n '
            json_data = {'system': 'This is a system prompt, please behave and help the user',
                         'conversation': [ { 'role': 'user', 'content': content } ] }
            response = requests.post(url=config['PLANETARYNAMES_PIPELINE_BRAIN_URL'],
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': 'Bearer %s' % config['PLANETARYNAMES_PIPELINE_BRAIN_API_TOKEN']},
                                     json=json_data)
            if response.status_code == 200:
                answer = response.json().get('text', '').strip()
                try:
                    answer = float(answer)
                    return min(1, max(0, answer))
                except ValueError:
                    return 0
            else:
                logger.error(f"From Brain status code {response.status_code}")

        return 0
