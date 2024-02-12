import requests
from tqdm import tqdm


class ConceptNet():

    @staticmethod
    def get_relation_url(entity: str) -> str:
        """
        function to generate query url for conceptnet

        Args:
            entity: name of the entity

        Returns:
            given entity's query url for conceptnet
        """
        return f'http://api.conceptnet.io/related/c/en/{entity}?filter=/c/en'


    @staticmethod
    def get_general_url(entity: str) -> str:
        """
        function to generate query url for conceptnet

        Args:
            entity: name of the entity

        Returns:
            given entity's query url for conceptnet
        """
        return f'http://api.conceptnet.io/c/en/{entity}?filter=/c/en&limit=1000'


    @staticmethod
    def get_relevance_url(ent1: str, ent2: str) -> str:
        """
        function to generate query url for conceptnet

        Args:
            ent1: name of the first entity
            ent2: name of the second entity

        Returns:
            given entities' query url for conceptnet
        """
        ent1 = '_'.join(ent1.split())
        ent2 = '_'.join(ent2.split())
        return f'http://api.conceptnet.io/relatedness?node1=/c/en/{ent1}&node2=/c/en/{ent2}'


    def get_related_entities(self, entity: str, n: int = 10) -> list[str]:
        """
        function to get entries from conceptnet

        Args:
            entity: name of the entity
            n: number of entries to return

        Returns:
            entries from conceptnet
        """
        url = self.get_relation_url(entity)
        concepts = requests.get(url).json()['related']
        concepts = [concept for concept in concepts if entity not in concept['@id']]
        concepts = sorted(concepts, key=lambda x: x['weight'], reverse=True, )[:10]
        concepts = [concept['@id'].split('/')[-1] for concept in concepts]
        return concepts
 

    def get_related_entity_location(self, entity: str) -> list:
        """
        function to get locations from conceptnet

        Args:
            entity: name of the entity

        Returns:
            locations from conceptnet
        """
        related_entities = self.get_related_entities(entity)

        location_list = []

        for entity in tqdm(related_entities, position=1, leave=False):
            result = self.get_entity_locations(entity)
            if result:
                for result_dict in result.values():
                    location_list.append(result_dict)

        return location_list

    
    def compute_relevance(self, ent1: str, ent2: str) -> float:
        """
        function to compute relevance between two entities 

        Args:
            ent1: name of the first entity 
            ent2: name of the second entity 

        Returns:
            relevance between two entities 
        """
        relevance_url = self.get_relevance_url(ent1, ent2)
        try:
            weight = requests.get(relevance_url).json()['value']
        except:
            weight = 0
        return float(weight)


    def get_entity_locations(self, entity: str, mover_preference: str, second_order_lookup: bool = False) -> list:
        """
        function to get locations from conceptnet

        Args:
            entity: name of the entity

        Returns:
            locations from conceptnet
        """
        url = self.get_general_url(entity)
        concepts = requests.get(url)

        location_list = []

        while concepts.ok:
            concepts = concepts.json()
            concept_edges = concepts['edges']

            location_concepts = [concept for concept in concept_edges if 'AtLocation' in concept['rel']['@id']]
            location_concepts = [concept for concept in location_concepts if entity not in concept['end']['label']]
            
            for location_concept in location_concepts:
                end_label = location_concept['end']['label']
                weight = location_concept['weight']

                # remove determinant
                if end_label.startswith(('a', 'an', 'the')):
                    end_label = ' '.join(end_label.split()[1:])

                location_list.append({
                    'end_label': end_label,
                    'weight': weight,
                    'place_len': len(end_label.split())
                })

            try:
                cur_view = concepts['view']
                next_url = cur_view['nextPage']
            except:
                break
            
            url = url.split('&')[0] + '&' + next_url.split('?')[-1]
            concepts = requests.get(url)

        # in case there is not LocateAt proprty available for the current entity,
        # perform another query to get the location of related entities
        if not location_list and second_order_lookup:
            
            print('First order lookup failed. Initiating second order lookup...')
            location_list = self.get_related_entity_location(entity)

        return location_list


def main():
    conceptnet = ConceptNet()
    location_dict = conceptnet.get_entity_locations('orange', second_order_lookup=True)


if __name__ == '__main__':
    main()
