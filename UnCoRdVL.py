import os
import re

import ctranslate2
import torch
from PIL import Image, ImageDraw

from estimators.object_detection import get_detection_model, get_transform
from vlbert.vqa.get_estimator import load_vlbert, get_prediction


class Node:

    def __init__(self, nid, F=None, N=None):
        self.nid = nid
        self.p = {}
        self.F = F
        self.N = N
        self.p_node = {}
        self.d_nodes = {}
        self.is_plural = False
        self.is_lower_node = False
        self.is_higher_node = False
        self.is_same_node = False
        self.nodeType = 'regular'
        self.nodes = []


class UnCoRdv2:

    def __init__(self, device):
        self.list_of_nodes = {}
        self.translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
        self.detector = get_detection_model(2)
        self.detector.load_state_dict(torch.load('./estimators/object_detection', map_location="cpu"))
        self.device = device
        self.estimator, self.transform, self.tokenizer = load_vlbert(ckpt_path='./vlbert/model/pretrained_model/vl'
                                                                               '-bert-base-prec.model', device=device)
        self.answer_vocab = ('gray', 'red', 'blue', 'green', 'brown', 'purple',
                             'cyan', 'yellow', 'small', 'large', 'rubber', 'metal',
                             'cube', 'sphere', 'cylinder', 'yes', 'no', '<unk>')

    def _nmt_seq2seq(self, question):
        """
        Google Neural Machine Translation
        returns question's graph sequence
        """
        with open('train_questions_none.graph') as f:
            graph_txt = f.read()
        graph_text_list = graph_txt.split('\n')
        ques_text = question['question']
        ques_text = re.sub(r'\?', '', ques_text)
        tokens = ques_text.split()
        graph_text = self.translator.translate_batch([tokens])

        print(' '.join(graph_text[0].hypotheses[0]))

        return ' '.join(graph_text[0].hypotheses[0])

        # return graph_text_list[question['question_index']]

    def _build_graph(self, graph_txt):
        """
        Returns list of graph nodes
        built from graph_txt sequence
        """
        COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
        SIZES = ['small', 'large']
        MATERIALS = ['rubber', 'metal']
        SHAPES = ['cube', 'sphere', 'cylinder']
        nodes_list = [node.strip() for node in graph_txt.split('<NewNode>')[1:]]
        for nid in range(len(nodes_list)):
            node_txt = nodes_list[nid].split()
            node = Node(nid + 1)
            # define object
            c = node_txt[1]
            node.p['shape'] = c
            # print(node_txt)
            # parsing relations
            if "<rd>" in node_txt or "<rp>" in node_txt:
                if "<rp>" in node_txt:
                    rp = p_id = -1
                    num_rps = node_txt.count("<rp>")
                    for i in range(num_rps):
                        rp = node_txt.index("<rp>", rp + 1)
                        p_rel = ''
                        for word in node_txt[rp + 1:]:
                            if word.isdigit():
                                p_id = word
                                break
                            else:
                                p_rel += word + ' '
                        node.p_node[p_id] = p_rel.strip()

                if "<rd>" in node_txt:
                    rd = d_id = -1
                    num_rds = node_txt.count("<rd>")
                    for i in range(num_rds):
                        rd = node_txt.index("<rd>", rd + 1)
                        d_rel = ''
                        for word in node_txt[rd + 1:]:
                            if word.isdigit():
                                d_id = word
                                break
                            else:
                                d_rel += word + ' '

                        node.d_nodes[d_id] = d_rel.strip()

            # parsing properties
            if "<p>" in node_txt:
                p_text = nodes_list[nid].split("<p>")[1:]
                p_text[-1] = p_text[-1].split('<')[0]
                properties = [p.strip() for p in p_text]
                for p in properties:
                    if p in COLORS:
                        node.p['color'] = p
                    elif p in SIZES:
                        node.p['size'] = p
                    elif p in MATERIALS:
                        node.p['material'] = p
                    elif p in SHAPES:
                        node.p['shape'] = p

            # print(node.p)

            # findig F
            if "<F>" in node_txt:
                node.F = node_txt[node_txt.index("<F>") + 1]

            # finding N
            if "<N>" in node_txt:
                node.N = node_txt[node_txt.index("<N>") + 1]

            if "<is_plural>" in node_txt:
                node.is_plural = True

            if "<lower_number_node>" in node_txt:
                node.is_lower_node = node_txt[node_txt.index("<lower_number_node>") + 1]

            if "<higher_number_node>" in node_txt:
                node.is_higher_node = node_txt[node_txt.index("<higher_number_node>") + 1]

            if "<same_number_node>" in node_txt:
                node.is_same_node = node_txt[node_txt.index("<same_number_node>") + 1]

            if "<nodeType>" in node_txt:
                node.nodeType = node_txt[node_txt.index("<nodeType>") + 1]
                if "<nodes>" in node_txt:
                    n_text = nodes_list[nid].split("<nodes>")[1:]
                    n_text[-1] = n_text[-1].split('<')[0]
                    node.nodes = [n.strip() for n in n_text]

            # print(node.nid)
            # print(node.F)
            # print(node.p_node)
            # print(node.d_nodes)
            self.list_of_nodes[nid + 1] = node

        # print(self.list_of_nodes)

    def get_answer(self, img_dir, question):

        if self.list_of_nodes:
            self.list_of_nodes.clear()

        img_id = question['image_index']

        graph_txt = self._nmt_seq2seq(question)
        self._build_graph(graph_txt)
        objects, scene = self._detect_objects(img_dir, img_id)
        answer = self._get_answer(1, objects, scene, candidate_objs=objects)

        return answer[1]

    def _get_answer(self, nid, objects, scene, visited_nodes=None, candidate_objs=None):
        """
        DFS Traversal (recursive)
        """
        if not visited_nodes:
            visited_nodes = {}
        # print(f'Checking node {nid}')
        answer = 'no'
        success = False
        cur_node = self.list_of_nodes[nid]
        cur_objects = []
        if cur_node.nodeType == 'superNode':
            candidate_objs.clear()
            for node in cur_node.nodes:
                for vis_node in visited_nodes[int(node)]:
                    if vis_node not in candidate_objs:
                        candidate_objs.append(vis_node)

        if candidate_objs:
            for candidate_obj in candidate_objs:
                if cur_node.p:
                    success, answer = self.check_properties(cur_node.nid, candidate_obj, scene)
                    if success:
                        # ("candidate is suitable")
                        cur_objects.append(candidate_obj)

        if cur_objects:
            success, answer = True, 'yes'
        else:
            success, answer = False, 'no'
        visited_nodes[nid] = cur_objects

        # print(f'Current objects: {cur_objects}')

        if cur_node.d_nodes or cur_node.p_node:
            if cur_node.d_nodes and cur_objects:
                cur_object = cur_objects[0]
                valid_objs = []
                for d_id in cur_node.d_nodes.keys():
                    if int(d_id) not in visited_nodes:
                        rel = cur_node.d_nodes[d_id]
                        for obj in objects:
                            obj_id = objects.index(obj)
                            cur_id = objects.index(cur_object)
                            if obj_id != cur_id:
                                # print(f'Checking properties of objects, related with {cur_node.nid}...')
                                success, answer = self.check_relations(cur_object, obj, cur_id, obj_id, rel, scene)
                            else:
                                continue

                            if success:
                                valid_objs.append(obj)

                        # print(f'Objects that are in suitable relation with node {nid}')
                        # print(valid_objs)
                        success, answer = self._get_answer(int(d_id), objects, scene, visited_nodes,
                                                           valid_objs)

            elif cur_node.p_node:
                # print('Checking parents')
                flag = False
                objects_to_remove = []
                for p_id in cur_node.p_node.keys():
                    # print(f'parent {p_id}')
                    if int(p_id) not in visited_nodes:
                        flag = True
                        rel = cur_node.p_node[p_id]
                        for cur_object in cur_objects:
                            # print(f"Searching parent object for {cur_object['color']} {cur_object['shape']}")
                            for obj in objects:
                                obj_id = objects.index(obj)
                                cur_id = objects.index(cur_object)

                                if obj_id != cur_id:
                                    # print('Checking properties...')
                                    success, answer = self.check_relations(obj, cur_object, obj_id, cur_id, rel, scene)
                                else:
                                    continue

                                if success:
                                    success, answer = self.check_properties(int(p_id), obj, scene)

                                if success:
                                    # print('found suitable parent object')
                                    break

                            if not success:
                                # print(f"No parent object for {cur_object['color']} {cur_object['shape']}")
                                objects_to_remove.append(cur_object)

                if not flag:
                    for i in self.list_of_nodes.keys():
                        if i not in visited_nodes:
                            success, answer = self._get_answer(i, objects, scene, visited_nodes, objects)
                            return success, answer
                else:
                    cur_objects = [obj for obj in cur_objects if obj not in objects_to_remove]
                    # print(f'Final objects: {cur_objects}')
                    if cur_objects:
                        success, answer = True, 'yes'
                    else:
                        success, answer = True, 'no'

        else:
            for i in self.list_of_nodes.keys():
                if i not in visited_nodes:
                    success, answer = self._get_answer(i, objects, scene, visited_nodes, objects)
                    return success, answer

        if cur_node.F:
            if cur_objects:
                success, answer = self.get_property_f(cur_node.nid, cur_objects[0], scene)

        elif cur_node.N:
            success, answer = self._count_objects(cur_objects)

        elif cur_node.is_lower_node or cur_node.is_higher_node or cur_node.is_same_node:
            # print('compare')
            low_node = cur_node.is_lower_node
            high_node = cur_node.is_higher_node
            same_node = cur_node.is_same_node
            # print(low_node)
            # print(high_node)
            if low_node:
                # print('lower_node')
                # print(visited_nodes.keys())
                if int(low_node) in visited_nodes.keys():
                    # print('getting answer')
                    # print(visited_nodes[2])
                    if cur_objects:
                        success, answer = self._compare_numbers(vis_obj=visited_nodes[int(low_node)],
                                                                cur_obj=cur_objects)
                    else:
                        success, answer = self._compare_numbers(vis_obj=visited_nodes[int(low_node)])
            elif high_node:
                # print('higher_node')
                if int(high_node) in visited_nodes.keys():
                    if cur_objects:
                        success, answer = self._compare_numbers(vis_obj=visited_nodes[int(high_node)], c_type='high',
                                                                cur_obj=cur_objects)
                    else:
                        success, answer = self._compare_numbers(vis_obj=visited_nodes[int(high_node)], c_type='high')
            else:
                if int(same_node) in visited_nodes.keys():
                    if cur_objects:
                        success, answer = self._compare_numbers(vis_obj=visited_nodes[int(same_node)], c_type='same',
                                                                cur_obj=cur_objects)
                    else:
                        success, answer = self._compare_numbers(vis_obj=visited_nodes[int(same_node)], c_type='same')

        return success, answer

    def check_properties(self, nid, obj, scene):

        cur_node = self.list_of_nodes[nid]
        success = False
        answer = 'no'
        boxes = [obj['box']]
        # boxes = desc[0]['boxes']
        # img.show()
        prediction = None
        for p_key in cur_node.p.keys():
            if cur_node.p[p_key] in ('object', 'item'):
                continue
            # print(f'Checking object {p_key}')
            prediction = get_prediction(scene, boxes, p_key, self.answer_vocab, self.tokenizer,
                                        self.estimator, self.transform)

            # self.visualize_results(scene, boxes, prediction)

            if prediction == cur_node.p[p_key]:
                continue
            else:
                return success, answer

        success = True
        answer = 'yes'

        return success, answer

    def get_property_f(self, nid, obj, scene):

        boxes = [obj['box']]

        cur_node = self.list_of_nodes[nid]
        answer = get_prediction(scene, [boxes], cur_node.F, self.answer_vocab, self.tokenizer,
                                self.estimator, self.transform)
        success = True

        # self.visualize_results(scene, boxes, answer)

        return success, answer

    def check_relations(self, cur_obj, obj, cur_id, obj_id, rel, scene):

        answer = 'no'
        success = False
        cur_box = cur_obj['box']
        box = obj['box']

        if 'same' in rel:
            boxes = [cur_box, box]
            # print('same check')
            same_p = rel.split()[1]
            cur_obj_prediction = get_prediction(scene, [cur_box], same_p, self.answer_vocab, self.tokenizer,
                                                self.estimator, self.transform)
            obj_prediction = get_prediction(scene, [box], same_p, self.answer_vocab, self.tokenizer,
                                            self.estimator, self.transform)

            if cur_obj_prediction == obj_prediction:
                success = True
                answer = 'yes'
            # self.visualize_results(scene, boxes, 'same' + same_p)
        else:
            boxes = [cur_box, box]
            prediction = get_prediction(scene, boxes, rel, self.answer_vocab, self.tokenizer,
                                        self.estimator, self.transform)

            if prediction == 'yes':
                success, answer = True, 'yes'

            # self.visualize_results(scene, boxes, prediction)
        return success, answer

    def _detect_objects(self, img_dir, img_id):
        """
        Faster RCNN object detection
        """

        transform = get_transform()
        img_path = list(sorted(os.listdir(img_dir)))[img_id]
        img = Image.open(img_dir + "/" + img_path).convert("RGB")
        # img.show()
        self.detector.eval()
        img_transformed = transform(img)
        prediction = self.detector(img_transformed.unsqueeze(0))[0]
        objects = [box.tolist() for box in prediction['boxes']]
        detections = []
        for box in objects:
            detection = {'box': box, 'img': img.crop(box)}
            detections.append(detection)

        return detections, img

    def _count_objects(self, cur_objects):

        if cur_objects:
            success, answer = True, str(len(cur_objects))
        else:
            success, answer = False, str(0)

        return success, answer

    def _compare_numbers(self, vis_obj=None, cur_obj=None, c_type='low'):

        success, answer = False, 'no'
        if cur_obj:
            cur_success, cur_answer = self._count_objects(cur_obj)
        else:
            cur_success, cur_answer = True, 0

        if vis_obj:
            vis_success, vis_answer = self._count_objects(vis_obj)
        else:
            vis_success, vis_answer = True, 0

        if cur_success and vis_success:
            # print(f'cur_objects number {cur_answer}')
            # print(f'vis_objects number {vis_answer}')
            if c_type == 'low':
                if int(cur_answer) > int(vis_answer):
                    success, answer = True, 'yes'
                else:
                    success, answer = False, 'no'
            elif c_type == 'high':
                if int(cur_answer) < int(vis_answer):
                    success, answer = True, 'yes'
                else:
                    success, answer = False, 'no'
            elif c_type == 'same':
                if int(cur_answer) == int(vis_answer):
                    success, answer = True, 'yes'
                else:
                    success, answer = False, 'no'

            return success, answer

    def visualize_results(self, image, boxes, answer):
        temp = image.copy()
        img = ImageDraw.Draw(temp)
        for i, box in enumerate(boxes):
            if i == 0:
                color = 'red'
            else:
                color = 'green'
            img.rectangle(box, outline=color)
            img.text(((box[2] + box[0]) / 2, (box[3] + box[1]) / 2), answer)
        temp.show()
