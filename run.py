import sys
import os
import csv
import re
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
from collections import Counter

from adsputils import setup_logging, load_config

import argparse

from adsplanetnamepipe import tasks
from adsplanetnamepipe.models import NamedEntityLabel
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


def get_default_filename(action_type: str) -> str:
    """
    get the default filename based on the action type and current timestamp

    :param action_type: str, the action type (e.g., export keywords or identified entities)
    :return: str, the generated output filename
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if action_type == PLANETARYNAMES_PIPELINE_ACTION.retrieve_knowledge_graph_keywords:
        return f"./knowledge_graph_keywords_{timestamp}.csv"
    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.retrieve_identified_entities:
        return f"./identified_entities_{timestamp}.csv"
    return ''


def output_identified_entities(output_file: str, identified_entities: List[Tuple[str, str, str, str, int, float, str]]) -> bool:
    """
    writes identified entities to a CSV file

    :param output_file: str, the file path to write the CSV
    :param identified_entities: list of tuples, each containing:
                                - bibcode: str, the bibcode of the entity
                                - target: str, the target entity
                                - feature_type: str, the feature type entity
                                - feature_name: str, the feature name entity
                                - feature_id: int, the feature ID from the database
                                - confidence_score: float, the confidence score
                                - date: str, the date in 'YYYY-MM-DD HH:MM:SS' format
    :return: bool, True if the file was written successfully, False otherwise
    """
    try:
        # Count the number of instances for each unique combination of bibcode, and feature ID
        counts = Counter((row[0], row[4]) for row in identified_entities)

        # Prepare the data for writing to the CSV
        aggregated_data = [
            {
                'bibcode': bibcode,
                'feature_id': feature_id,
                'feature_name': next(row[3] for row in identified_entities if row[0] == bibcode and row[4] == feature_id),
                'feature_type': next(row[2] for row in identified_entities if row[0] == bibcode and row[4] == feature_id),
                'target': next(row[1] for row in identified_entities if row[0] == bibcode and row[4] == feature_id),
                'num_instances': count
            }
            for (bibcode, feature_id), count in counts.items()
        ]

        file_exists = os.path.isfile(output_file)
        with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # write the header only if the file is new
            if not file_exists:
                writer.writerow(['Feature ID', 'Feature Name', 'Feature Type', 'Target', 'Bibcode', 'Number of Instances'])

            # Write each identified entity to the file
            for entity in aggregated_data:
                writer.writerow([
                    entity['feature_id'],
                    entity['feature_name'],
                    entity['feature_type'],
                    entity['target'],
                    entity['bibcode'],
                    entity['num_instances']
                ])
            return True

    except Exception as e:
        logger.error(f"Failed to write identified entities to '{output_file}': {e}")
        return False


def output_knowledge_graph_keywords(output_file: str, feature_name: str, feature_type: str, target: str, label: str, keywords: List[str]) -> bool:
    """
    writes the keywords to a CSV file, appending to the file if it already exists

    :param output_file: str, the file path to write the CSV
    :param feature_name: str, the feature name entity
    :param feature_type: the feature type entity
    :param target: str, the target entity
    :param label: str, named entity label (e.g., planetary, unknown)
    :param keywords: list of keywords to write to the file
    :return: bool, True if writing is successful, False otherwise
    """
    try:
        file_exists = os.path.isfile(output_file)
        with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # write the header only if the file is new
            if not file_exists:
                writer.writerow(['Feature Name', 'Feature Type', 'Target', 'Label', 'Keywords'])

            for keyword in keywords:
                writer.writerow([feature_name, feature_type, target, label, keyword])

        return True
    except Exception as e:
        logger.error(f"Failed to write keywords to CSV: {str(e)}")
        return False


def process_a_feature_name(feature_name: str, target: str, feature_type: str, action_type: str,
                           keyword: str, timestamp: datetime, output_file: str, label: str):
    """
    processes a single feature name based on the provided action type

    :param feature_name: str, the name of the feature to be processed
    :param target: str, the current target entity (e.g., Moon, Mars)
    :param feature_type: str, the feature type (e.g., Crater)
    :param action_type: str, the action to perform (e.g., collect, identify, etc.)
    :param keyword: str, the keyword to be used for keyword-related actions, otherwise it is empty
    :param timestamp: datetime, timestamp for identifying or processing entities
    :param output_file: str, the file name for data export
    :param label: str, label to specify getting planetary or non-planetary keywords (ie, planetray or unknown)
    """
    entity_args = EntityArgs(target=target,
                             feature_type=feature_type,
                             feature_type_plural=app.get_plural_feature_type_entity(feature_type),
                             feature_name=feature_name,
                             context_ambiguous_feature_names=app.get_context_ambiguous_feature_name(feature_name),
                             multi_token_containing_feature_names=app.get_multi_token_containing_feature_name(feature_name),
                             name_entity_labels=app.get_named_entity_label(),
                             timestamp=timestamp,
                             all_targets=app.get_target_entities())

    # these actions go through queue
    if action_type in [PLANETARYNAMES_PIPELINE_ACTION.collect,
                       PLANETARYNAMES_PIPELINE_ACTION.identify,
                       PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
        # serialize before queueing
        the_task = {'action_type': action_type.value, 'args': entity_args.toJSON()}
        tasks.task_process_planetary_nomenclature.delay(the_task)

    # the following five actions is applied to knowledge graph (ie, remove most recent record, remove all all but most recent records,
    # remove keywords, add keywords, export keywords)

    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_the_most_recent:
        KBH_rows_deleted, KB_rows_deleted = app.remove_most_recent_knowledge_base_records(feature_name_entity=feature_name,
                                                                                          target_entity=target)
        logger.info(
            f"Removed the most recent records of knowledge graph for feature name `{feature_name}` and target `{target}`.\n"
            f"Deleted {KBH_rows_deleted} rows from knowledge_base_history and {KB_rows_deleted} rows from knowledge_base.")

    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_all_but_last:
        KBH_rows_deleted, KB_rows_deleted = app.remove_all_but_most_recent_knowledge_base_records(feature_name_entity=feature_name,
                                                                                                  target_entity=target)
        logger.info(
            f"Removed all but the most recent records of knowledge graph for feature name `{feature_name}` and target `{target}`.\n"
            f"Deleted {KBH_rows_deleted} rows from knowledge_base_history and {KB_rows_deleted} rows from knowledge_base.\n")

    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.add_keyword_to_knowledge_graph:
        if keyword:
            rows_updated = app.append_to_knowledge_base_keywords(feature_name_entity=feature_name,
                                                                 target_entity=target, keyword=keyword)
            logger.info(f"{rows_updated} rows updated by adding the keyword.")
        else:
            logger.info('Keyword is a required parameter for this action! Terminating!')

    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_keyword_from_knowledge_graph:
        if keyword:
            rows_updated = app.remove_from_knowledge_base_keywords(feature_name_entity=feature_name,
                                                                   target_entity=target, keyword=keyword)
            logger.info(f"{rows_updated} rows updated by removing the keyword.")
        else:
            logger.info('Keyword is a required parameter for this action! Terminating!')

    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.retrieve_knowledge_graph_keywords:
        if not output_file:
            output_file = get_default_filename(action_type)

        named_entity_label = NamedEntityLabel.verify_label(label)
        keywords = app.get_knowledge_base_keywords(feature_name_entity=feature_name,
                                                   feature_type_entity=feature_type,
                                                   target_entity=target,
                                                   named_entity_label=named_entity_label)
        if keywords:
            if output_knowledge_graph_keywords(output_file, feature_name, feature_type, target, named_entity_label, keywords):
                logger.info(f"Added {len(keywords)} keywords for feature name '{feature_name}', feature type '{feature_type}', target '{target}', and label '{named_entity_label}' to '{output_file}'.")
            else:
                logger.error(f"Failed to add {len(keywords)} keywords for feature name '{feature_name}', feature type '{feature_type}', target '{target}', and label '{named_entity_label}' to '{output_file}'.")
        else:
            logger.info(
                f"No keywords to add for feature name '{feature_name}', feature type '{feature_type}', target '{target}', and label '{named_entity_label}'.")


def read_updated_usgs_gazetteer(csv_file_path: str) -> List[Dict[str, str]]:
    """
    reads the updated usgs entities
    containing five columns: Feature_ID,Clean_Feature_Name,Target,Feature_Type,Approval_Date,Approval_Status

    :param csv_file_path: path to the CSV file.
    :return: list of dictionaries, each containing data from one row.
    """
    year_pattern = re.compile(r'\b\d{4}\b')
    data = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            # there are lots of white space in the header line coming from the Planetary Name website
            # Feature ID	Clean          Feature Name	Target	Feature Type	Approval          Status	Approval Date
            # skip their header
            old_header = next(reader)
            # to use these as column headers
            new_header = ['entity_id', 'feature_name', 'target', 'feature_type', 'approval_status', 'approval_date']
            for row in reader:
                # transform keys first
                row = {new_key: row[old_key] for new_key, old_key in zip(new_header, old_header)}
                # now process the row
                if row['approval_status'].lower() == 'approved':
                    # extract year from `approval_date`
                    match = year_pattern.search(row['approval_date'])
                    row['approval_date'] = match.group(0) if match else row['approval_date']
                    # split `feature_type` to have separate singular and plural (if any)
                    row['feature_type'], row['feature_type_plural'] = [x.strip() for x in (row['feature_type'].split(',') + [''])[:2]]
                    data.append(row)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file_path}. Please check the path and try again.")
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
    return data


def import_usgs_update(usgs_update_file: str):
    """
    import the USGS Gazetteer update by reading the input file, identifying new entity IDs,
    and updating the necessary database tables.

    :param usgs_update_file: str, path to the USGS update CSV file
    """
    data = read_updated_usgs_gazetteer(usgs_update_file)
    if data:
        entity_id = {int(row['entity_id']) for row in data}
        entity_ids = set(app.get_entity_ids())
        new_entity_id = list(entity_id - entity_ids)
        # extract entries for the new_entity_id and send to be checked for any new target or feature_type
        # to be inserted to the database
        new_entries = [row for row in data if int(row['entity_id']) in new_entity_id]
        tables_updated = app.add_new_usgs_entities(new_entries)
        if tables_updated:
            logger.info("Imported USGS Gazetteer successfully.")
        else:
            logger.error("Failed to import USGS Gazetteer.")


def get_date(days: int) -> datetime:
    """
    calculates a target date based on the number of days provided
    if days is 0, it returns the earliest possible datetime (datetime.min).

    :param days: int, the number of days to subtract from the current date. Can be 0 or a positive integer.
    :return: datetime, a datetime object representing either the target date or the earliest possible datetime if days is 0.
    """
    if not days or days == 0:
        return datetime.min

    target_date = datetime.now().date() - timedelta(days=days)
    target_datetime = datetime.combine(target_date, datetime.now().time())
    return target_datetime


def process_timestamp(action_type: str, timestamp_arg: str, default_timestamp: datetime) -> datetime:
    """
    processes the timestamp argument based on the provided action type

    :param action_type: str, the type of action being performed (e.g., identify, end_to_end, etc.)
    :param timestamp_arg: str, the timestamp argument provided (if any), should be in 'YYYY-MM-DD' format
    :param :param default_timestamp: datetime, the default timestamp to be used if no valid input is provided or if the action type does not support timestamp processing
    :return: datetime, the processed timestamp value
    """
    if action_type in [PLANETARYNAMES_PIPELINE_ACTION.identify, PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
        if timestamp_arg:
            try:
                # verify the format of the timestamp
                timestamp = datetime.strptime(timestamp_arg, '%Y-%m-%d')
            except ValueError:
                logger.error(f"The timestamp '{timestamp_arg}' is not in the correct format YYYY-MM-DD. Default 2000-01-01 is used.")
                timestamp = default_timestamp
        else:
            timestamp = default_timestamp
    else:
        logger.info("`timestamp` argument is only applicable when action=identify. Ignoring the timestamp using default 2000-01-01.")
        timestamp = default_timestamp

    return timestamp


def verify_arguments(args: argparse) -> Tuple[str, str, List[str]]:
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


def parse_arguments() -> argparse.Namespace:
    """
    parses the command-line arguments for the script

    :return: argparse.Namespace, an object containing the parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Identify Planetary Nomenclature')
    parser.add_argument('-a', '--action', help='action to perform: collect, identify, end_to_end, etc.')
    parser.add_argument('-t', '--target', help='target (e.g., Moon, Mars). capitalized.')
    parser.add_argument('-f', '--feature_type', help='feature type (e.g., Crater). capitalized and singular.')
    parser.add_argument('-n', '--feature_name', help='feature name (e.g., Apollo). capitalized.')
    parser.add_argument('-k', '--keyword', help='knowledge graph keyword to add/remove/export.')
    parser.add_argument('-c', '--confidence_score', help='optional: filter by minimum confidence score.')
    parser.add_argument('-d', '--days', help='optional: filter by days for retrieve_identified_entities action.')
    parser.add_argument('-s', '--timestamp', help='optional: specify date in YYYY-MM-DD format for identification actions.')
    parser.add_argument('-u', '--usgs_update', help='CSV file for USGS data update.')
    parser.add_argument('-o', '--output_file', help='optional: specify file name for data export. If omitted, defaults to `keywords_export_<timestamp>.csv` or `identified_entities_<timestamp>.csv`, saved in the current directory.')
    parser.add_argument('-l', '--label', help='optional: specify label of the knowledge graph keywords to export (e.g, planetary or unknown).')
    return parser.parse_args()


# python run.py -t Mars -f "Albedo Feature" -a collect
# python run.py -t Mars -n Amenthes -a collect
# python run.py -t Mars -n Amenthes -a remove_all_but_last
# python run.py -t Mars -n Amenthes -a remove_the_most_recent
# python run.py -t Mars -n Amenthes -a add_keyword_to_knowledge_graph -k basin -t Moon -f Apollo
# python run.py -t Mars -n Amenthes -a remove_keyword_from_knowledge_graph
# python run.py -t Mars -n Amenthes -a retrieve_identified_entities -c 0.75
# python run.py -t Mars -n Amenthes -a retrieve_knowledge_graph_keywords -l planetary
# TODO: add a command to run all the feature types for all celestial bodies for collect step
# TODO: add a command to run all the feature names since d days ago
# python run.py -a update_database_with_usgs_entities -u updated_usgs_terms.csv

# Main entry point of the script.
# Sets up argument parsing, processes the arguments, and executes the appropriate action based on the provided command-line arguments.
if __name__ == '__main__':

    args = parse_arguments()
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
        current_target, current_feature_type, current_feature_names = verify_arguments(args)
        current_feature_names = current_feature_names if len(current_feature_names) > 0 else ['']
        for feature_name in current_feature_names:
            results = app.get_named_entity_bibcodes(feature_name_entity=feature_name,
                                                    feature_type_entity=current_feature_type,
                                                    target_entity=current_target,
                                                    confidence_score=args.confidence_score,
                                                    date=get_date(args.days))


            if results:
                output_file = args.output_file if args.output_file else get_default_filename(action_type)
                if output_identified_entities(output_file, results):
                    logger.info(f"Added {len(results)} identified entities for feature name '{feature_name}', feature type '{current_feature_type}', target '{current_target}' to '{output_file}'.")
                else:
                    logger.error(f"Failed to add {len(results)} identified entities for feature name '{feature_name}', feature type '{current_feature_type}', target '{current_target}' to '{output_file}'.")
            else:
                logger.info(f"No identified entities for feature name '{feature_name}', feature type '{current_feature_type}', target '{current_target}'.")

    # this action requires one parameter: the csv file extracted info from usgs recently
    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.update_database_with_usgs_entities:
        if args.usgs_update:
            import_usgs_update(args.usgs_update)
        else:
            logger.error("CSV file for USGS data update is missing. Please provide the CSV file using the '--usgs_update' argument.")
            sys.exit(1)

    # the rest of the actions require the target and either feature name or feature type arguments
    else:
        if args.target:
            timestamp = process_timestamp(action_type, args.timestamp, config['PLANETARYNAMES_PIPELINE_DEFAULT_TIMESTAMP'])
            current_target, current_feature_type, current_feature_names = verify_arguments(args)
            # optional params
            if current_target and (current_feature_type or current_feature_names):
                # process one to many feature names
                # one is when feature name is entered, while many is when feature type is entered
                for feature_name in current_feature_names:
                    process_a_feature_name(feature_name, current_target, current_feature_type, action_type, args.keyword,
                                           timestamp, args.output_file, args.label)
            else:
                logger.info('Either valid feature type (-f) or valid feature name (-n) is needed for processing! Terminating!')
                sys.exit(1)
        else:
            logger.info('The target (-t) is needed for processing! Terminating!')
            sys.exit(1)

    sys.exit(0)
