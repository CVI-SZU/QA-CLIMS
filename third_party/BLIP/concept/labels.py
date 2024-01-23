from nltk.corpus import wordnet
import requests
from pprint import pprint

voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
              'dog',
              'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']


# find synonyms of a word from wordnet
def find_synonyms_wornet(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms))


def get_request(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'Error: {response.status_code}')
    except Exception as e:
        print(e)
        exit(-1)


class ConceptNet:
    def __init__(self):
        self.api_root = 'https://api.conceptnet.io/'

    def get_word_id(self, word):
        url = self.api_root + f'c/en/{word}'
        data = get_request(url)
        return data['@id']

    def query_c_edges(self, word_id, rel_id=None, as_start=True, as_end=False, limit=100):
        assert as_start != as_end
        assert as_start or as_end
        url = self.api_root + 'query?'
        if as_start:
            url = f'{url}start={word_id}'
        else:
            url = f'{url}end={word_id}'
        if rel_id is not None:
            url = f'{url}&rel={rel_id}'
        url = f'{url}&limit={limit}'
        data = get_request(url)
        edges = data['edges']
        return edges

    def format_edges(self, edges, as_start, weight_threshold=0.5):
        formatted_edges = []
        for edge in edges:
            if edge['weight'] < weight_threshold:
                continue
            if as_start:
                target_label = edge['end']['label']
            else:
                target_label = edge['start']['label']
            formatted_edges.append({
                'target': target_label,
                'weight': edge['weight']
            })
        # sort by weight big to small
        formatted_edges = sorted(formatted_edges, key=lambda x: x['weight'], reverse=True)
        return formatted_edges


# find synonyms of a word from ConceptNet
def find_synonyms_conceptnet(word):
    concept_net = ConceptNet()
    rel_ids = ['/r/RelatedTo', '/r/PartOf']
    as_start = [True, False]
    # weight_thresholds = [5, 1.0]
    weight_thresholds = [1.0, 1.0]
    topk = 10
    for rel_id, st, th in zip(rel_ids, as_start, weight_thresholds):
        word_id = concept_net.get_word_id(word)
        edges = concept_net.query_c_edges(word_id, rel_id, as_start=st, as_end=not st, limit=200)
        edges = concept_net.format_edges(edges, as_start=st, weight_threshold=th)
        print(f'>>{rel_id} as_start={st} th={th} topk={topk}')
        pprint(edges[:topk])


for label in voc_labels:
    print(f'\n>>>>{label}')
    print('[[[ wordnet ]]]')
    print(find_synonyms_wornet(label))
    print('[[[ conceptnet ]]]')
    find_synonyms_conceptnet(label)

