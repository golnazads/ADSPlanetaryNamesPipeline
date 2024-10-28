from typing import List, Tuple, Dict
from collections import defaultdict

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

import networkx as nx

from adsplanetnamepipe.utils.common import EntityArgs


class KnowledgeGraph():

    """
    a class that builds and queries a knowledge graph based on keywords from scientific texts

    this class creates a graph representation of keyword relationships, with special
    emphasis on certain keywords, and provides methods to query the graph for relevance scores
    """

    def __init__(self, args: EntityArgs, keywords: List[List[str]], special_keywords: List[List[str]]):
        """
        initialize the KnowledgeGraph class

        :param args: configuration arguments containing feature name and other settings
        :param keywords: list of keyword lists extracted from texts
        :param special_keywords: list of special keyword lists with higher weights
        """
        self.args = args
        self.score_format = '%.{}f'.format(config['PLANETARYNAMES_PIPELINE_FORMAT_SIGNIFICANT_DIGITS'])
        self.graph = nx.Graph()

        self.build_graph(keywords, special_keywords)

    def build_graph(self, keywords_list: List[List[str]], special_keywords_list: List[List[str]]):
        """
        build the knowledge graph from the provided keywords and special keywords

        :param keywords_list: list of keyword lists
        :param special_keywords_list: list of special keyword lists with higher weights
        :return:
        """
        self.keyword_counts, self.pair_counts = self.count_keywords(keywords_list)
        self.special_keyword_counts, self.special_pair_counts = self.count_keywords(special_keywords_list)

        if len(self.keyword_counts) > 0:
            self.create_graph()
        else:
            self.graph = None

    def create_graph(self):
        """
        create the graph structure based on keyword and pair counts

        :return:
        """
        for (keyword_counts, pair_counts, weight) in zip([self.keyword_counts, self.special_keyword_counts],
                                                         [self.pair_counts, self.special_pair_counts],
                                                         [1, 2]):
            keyword_list = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            pair_list = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
            # weights for regular keywords are 1, for special keywords are 2
            self.add_nodes_and_edges(keyword_list, pair_list, weight)

    def count_keywords(self, keywords_list: List[List[str]]) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """
        count the occurrences of individual keywords and keyword pairs

        :param keywords_list: list of keyword lists
        :return: tuple of dictionaries containing keyword counts and pair counts
        """
        keyword_counts = defaultdict(int)
        pair_counts = defaultdict(int)

        for keywords in keywords_list:
            for keyword in keywords:
                keyword_counts[keyword] += 1

                for other_keyword in keywords:
                    if keyword != other_keyword:
                        pair_counts[(keyword, other_keyword)] += 1
        return keyword_counts, pair_counts

    def add_nodes_and_edges(self, keyword_list: List[Tuple[str, int]], pair_list: List[Tuple[Tuple[str, str], int]], weight: int):
        """
        add nodes and edges to the graph based on keyword and pair counts

        :param keyword_list: list of tuples containing keywords and their counts
        :param pair_list: list of tuples containing keyword pairs and their counts
        :param weight: weight factor for the edges
        :return:
        """
        for keyword, count in keyword_list:
            self.graph.add_node(keyword)
            self.graph.add_edge(keyword, self.args.feature_name, weight=count*weight)

        for (keyword1, keyword2), count in pair_list:
            if self.graph.has_node(keyword1) and self.graph.has_node(keyword2*weight):
                self.graph.add_edge(keyword1, keyword2, weight=count)

    def query_path(self, keyword: str) -> float:
        """
        query the graph for the path weight between a keyword and the feature name

        :param keyword: the keyword to query
        :return: float the average weight of the path
        """
        if not self.graph.has_node(keyword):
            return 0

        try:
            path = nx.shortest_path(self.graph, keyword, self.args.feature_name)
            path_weights = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.graph.has_edge(u, v):
                    weight = self.graph[u][v].get('weight', 0)
                    path_weights.append(weight)

            avg_weight = sum(path_weights) / len(path_weights) if path_weights else 0

            return avg_weight

        except nx.NetworkXNoPath:
            return 0

    def forward(self, keywords: List[str]) -> float:
        """
        calculate the average path weight for a list of keywords

        :param keywords: list of keywords to query
        :return: float the average path weight, or -1 if the graph is empty
        """
        # if no data was available to setup the knowledge graph,
        # send the indication with socre = -1
        if not self.graph:
            return -1

        path_weights = 0
        for keyword in keywords:
            path_weights += self.query_path(keyword)

        score = float(self.score_format % (path_weights / len(keywords)))
        return score
