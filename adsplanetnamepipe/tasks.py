from adsplanetnamepipe import app as app_module
from kombu import Queue

import os

from adsplanetnamepipe.utils.common import PLANETARYNAMES_PIPELINE_ACTION, PlanetaryNomenclatureTask
from adsplanetnamepipe.collect import CollectKnowldegeBase
from adsplanetnamepipe.identify import IdentifyPlanetaryEntities

from adsputils import load_config

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
config = load_config(proj_home=proj_home)
app = app_module.ADSPlanetaryNamesPipelineCelery('planetary-names-pipeline',
                                                 proj_home=proj_home,
                                                 local_config=globals().get('local_config', {}))

app.conf.CELERY_QUEUES = (
    Queue('task_process_planetary_nomenclature', app.exchange, routing_key='task_process_planetary_nomenclature'),
)

logger = app.logger


class FailedRequest(Exception):
    """
    Failed to connect to reference service.
    """
    pass


@app.task(queue='task_process_planetary_nomenclature', max_retries=config['MAX_QUEUE_RETRIES'])
def task_process_planetary_nomenclature(the_task: PlanetaryNomenclatureTask) -> bool:
    """
    processes planetary nomenclature tasks based on the provided action type

    handles the two main actions:
        1. collecting data for knowledge base setup
        2. identifying and labeling entities

    :param the_task: PlanetaryNomenclatureTask, A typed dictionary containing:
                     - 'action_type': PLANETARYNAMES_PIPELINE_ACTION enum value
                     - 'args': EntityArgs object containing task arguments
    :return: bool, returns True if the task is processed successfully, False otherwise
    """
    try:
        # either: action to collect data to setup KB graph
        if the_task['action_type'] in [PLANETARYNAMES_PIPELINE_ACTION.collect,
                                       PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
            knowledge_base_records = CollectKnowldegeBase(the_task['args']).collect()
            if knowledge_base_records:
                return bool(app.insert_knowledge_base_records(knowledge_base_records))

            logger.info(f"No knowledge base records found for: {the_task['args'].feature_name}/{the_task['args'].feature_type}/{the_task['args'].target}")
            return False

        # or: action to identify and label entities
        if the_task['action_type'] in [PLANETARYNAMES_PIPELINE_ACTION.identify,
                                       PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
            keywords_positive = app.get_knowledge_base_keywords(the_task['args'].feature_name,
                                                                the_task['args'].feature_type,
                                                                the_task['args'].target,
                                                                the_task['args'].name_entity_labels[0]['label'])
            keywords_negative = app.get_knowledge_base_keywords(the_task['args'].feature_name,
                                                                the_task['args'].feature_type,
                                                                the_task['args'].target,
                                                                the_task['args'].name_entity_labels[1]['label'])
            named_entity_records = IdentifyPlanetaryEntities(the_task['args'], keywords_positive,
                                                             keywords_negative).identify()
            if named_entity_records:
                return bool(app.insert_named_entity_records(named_entity_records))

            logger.info(f"No records identified for: {the_task['args'].feature_name}/{the_task['args'].feature_type}/{the_task['args'].target}")
            return False

        logger.error(f"Unhandled action: {the_task['action_type']}")
        return False

    except KeyError as e:
        logger.error(f"KeyError in task_process_planetary_nomenclature: {str(e)}")
        return False


# dont know how to unittest this part
# this (app.start()) the only line that is not unittested
# and since i want all modules to be 100% covered,
# making this line not be considered part of coverage
if __name__ == '__main__':    # pragma: no cover
    app.start()