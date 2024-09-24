import sys
import os
from typing import List, Tuple
from datetime import datetime, timedelta

from adsputils import setup_logging, load_config

import argparse

from adsplanetnamepipe import tasks
from adsplanetnamepipe.utils.common import PLANETARYNAMES_PIPELINE_ACTION, EntityArgs

proj_home = os.path.realpath(os.path.dirname(__file__))
config = load_config(proj_home=proj_home)

app = tasks.app
logger = setup_logging('run.py')


def map_input_param_to_action_type(input_param: str) -> PLANETARYNAMES_PIPELINE_ACTION:
    """
    maps an input parameter to a corresponding action type from the PLANETARYNAMES_PIPELINE_ACTION enum

    :param input_param: input parameter to be mapped to an action type
    :return: PLANETARYNAMES_PIPELINE_ACTION, corresponding PLANETARYNAMES_PIPELINE_ACTION enum value
    """
    try:
        return PLANETARYNAMES_PIPELINE_ACTION[input_param]
    except (KeyError, ValueError):
        return PLANETARYNAMES_PIPELINE_ACTION.invalid


def verify_args(args: argparse) -> Tuple[str, str, List[str]]:
    """
    verifies and processes the command-line arguments, returning the target, feature type, and feature names

    :param args: parsed command-line arguments object
    :return: Tuple[str, str, List[str]], A tuple containing (target, feature_type, feature_names)
    """
    if args.target and args.feature_type:
        return args.target, args.feature_type, app.get_feature_name_entities(args.target, args.feature_type)
    if args.target and args.feature_name:
        return args.target, app.get_feature_type_entity(args.target, args.feature_name), [args.feature_name]
    return args.target, args.feature_type, [args.feature_name]


def get_date(days: int) -> datetime:
    """
    Calculates a target date based on the number of days provided.
    If days is 0, it returns the earliest possible datetime (datetime.min).

    :param days: int, The number of days to subtract from the current date. Can be 0 or a positive integer.
    :return: datetime, A datetime object representing either the target date or the earliest possible datetime if days is 0.
    """
    if days == 0:
        return datetime.min

    target_date = datetime.now().date() - timedelta(days=days)
    target_datetime = datetime.combine(target_date, datetime.now().time())
    return target_datetime

# python run.py -t Mars -f "Albedo Feature" -a collect
# python run.py -t Mars -n Amenthes -a collect
# python run.py -t Mars -n Amenthes -a remove_all_but_last
# python run.py -t Mars -n Amenthes -a remove_the_most_recent
# python run.py -t Mars -n Amenthes -a add_keyword_to_knowledge_graph -k basin -t Moon -f Apollo
# python run.py -t Mars -n Amenthes -a remove_keyword_from_knowledge_graph
# python run.py -t Mars -n Amenthes -a retrieve_identified_entities -c 0.75
# TODO: add a command to run all the feature types for all celestial bodies for collect step
# TODO: add a command to run all the feature names since d days ago

# Main entry point of the script.
# Sets up argument parsing, processes the arguments, and executes the appropriate action based on the provided command-line arguments.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify Planetary Nomenclature')
    parser.add_argument('-a', '--action', help='one of the followings: collect, identify, end_to_end (collect followed by identify).')
    parser.add_argument('-t', '--target', help='target (ie, Moon, Mars, etc) is required. Has to be capitalized.')
    parser.add_argument('-f', '--feature_type', help='feature type (ie, Crater or Albedo Feature), is one of the options, either feature type or feature name are required, but no both. Has to be capitalized and in singular form.')
    parser.add_argument('-n', '--feature_name', help='feature name (ie Apollo, Atlas), is one of the options, either feature type or feature name are required, but no both. Has to be capitalized.')
    parser.add_argument('-k', '--keyword', help='knowledge graph keyword to add or remove manually.')
    parser.add_argument('-c', '--confidence_score', help='(optional) only applicable for action=retrieve_identified_entities, if specified only the identified entities with confidence score >= this score are returned.')
    parser.add_argument('-d', '--days', help='(optional) only applicable for action=retrieve_identified_entities, if specified only the entities identified in the past many days are returned.')
    args = parser.parse_args()
    if args.action:
        action_type = map_input_param_to_action_type(args.action)
        if action_type == PLANETARYNAMES_PIPELINE_ACTION.invalid:
            logger.info(f"Invalid action arg `{args.action}`! Terminating!")
            sys.exit(1)
    else:
        logger.info('The action arg (-a) is needed for processing! Terminating!')
        sys.exit(1)

    # the only action command with no required parameter
    if action_type == PLANETARYNAMES_PIPELINE_ACTION.retrieve_identified_entities:
        current_target, current_feature_type, current_feature_names = verify_args(args)
        current_feature_names = current_feature_names if len(current_feature_names) > 0 else ['']
        for feature_name in current_feature_names:
            results = app.get_named_entity_bibcodes(feature_name, current_feature_type, current_target, args.confidence_score, get_date(args.days))
            # TODO: find out how the user wants this results outputted
            if results:
                logger.info(f"Total entities fetched: {len(results)}")
    else:
        if args.target:
            current_target, current_feature_type, current_feature_names = verify_args(args)
            if current_target and (current_feature_type or current_feature_names):
                # process one to many feature names
                # one is when feature name is entered, while many is when feature type is entered
                for feature_name in current_feature_names:
                    entity_args = EntityArgs(target=current_target,
                                             feature_type=current_feature_type,
                                             feature_type_plural=app.get_plural_feature_type_entity(current_feature_type),
                                             feature_name=feature_name,
                                             context_ambiguous_feature_names=app.get_context_ambiguous_feature_name(feature_name),
                                             multi_token_containing_feature_names=app.get_multi_token_containing_feature_name(feature_name),
                                             name_entity_labels=app.get_named_entity_label(),
                                             all_targets=app.get_target_entities())
                    # actions that needs to go through queue
                    if action_type in [PLANETARYNAMES_PIPELINE_ACTION.collect,
                                       PLANETARYNAMES_PIPELINE_ACTION.identify,
                                       PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
                        the_task = {'action_type': action_type, 'args': entity_args}
                        tasks.task_process_planetary_nomenclature.delay(the_task)

                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_the_most_recent:
                        KBH_rows_deleted, KB_rows_deleted = app.remove_most_recent_knowledge_base_records(feature_name_entity=feature_name,
                                                                                                          target_entity=current_target)
                        logger.info(f"Removed the most recent records of knowledge graph for feature name `{feature_name}` and target `{current_target}`.\n"
                                    f"Deleted {KBH_rows_deleted} rows from knowledge_base_history and {KB_rows_deleted} rows from knowledge_base.")
                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_all_but_last:
                        KBH_rows_deleted, KB_rows_deleted = app.remove_all_but_most_recent_knowledge_base_records(feature_name_entity=feature_name,
                                                                                                                  target_entity=current_target)
                        logger.info(f"Removed all but the most recent records of knowledge graph for feature name `{feature_name}` and target `{current_target}`.\n"
                                    f"Deleted {KBH_rows_deleted} rows from knowledge_base_history and {KB_rows_deleted} rows from knowledge_base.\n")
                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.add_keyword_to_knowledge_graph:
                        if args.keyword:
                            rows_updated = app.append_to_knowledge_base_keywords(feature_name_entity=feature_name,target_entity=current_target, keyword=args.keyword)
                            logger.info(f"{rows_updated} rows updated by adding the keyword.")
                        else:
                            logger.info('Keyword (-k) is a required parameter for this action! Terminating!')
                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_keyword_from_knowledge_graph:
                        if args.keyword:
                            rows_updated = app.remove_from_knowledge_base_keywords(feature_name_entity=feature_name,target_entity=current_target, keyword=args.keyword)
                            logger.info(f"{rows_updated} rows updated by removing the keyword.")
                        else:
                            logger.info('Keyword (-k) is a required parameter for this action! Terminating!')
            else:
                logger.info('Either valid feature type (-f) or valid feature name (-n) is needed for processing! Terminating!')
                sys.exit(1)
        else:
            logger.info('The target (-t) is needed for processing! Terminating!')
            sys.exit(1)
    sys.exit(0)