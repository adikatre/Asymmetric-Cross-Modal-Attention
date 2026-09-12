"""Offline checks of notebook definitions; no VQA data or pretrained downloads.

Run with: python -m unittest discover -s tests -v
"""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel, RobertaConfig, RobertaModel


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb'


def definitions():
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [''.join(c.get('source', [])) for c in notebook['cells']]
    ns = {'__name__': 'vqa_notebook_test'}
    exec(cells[5].split('# Fast tokenizers')[0], ns)
    ns.update(IMAGE_ENCODER_NAME='fixture', VISION_TRAINING_MODE='full', IMAGE_SIZE=32,
              IMAGE_PATCH_SIZE=16, IMAGE_SEQ_LEN=5, EMBED_DIM=16, NUM_HEADS=4, NUM_ANSWERS=3,
              DROPOUT=0.1, FREEZE_ENCODERS=True, SEED=7, SEEDS=(7, 42, 123),
              BATCH_SIZE=2, NUM_WORKERS=0, EPOCHS=3, GRAD_ACCUM_STEPS=1,
              MAX_GRAD_NORM=1.0, LEARNING_RATE=2e-4, ENCODER_LR=2e-6, WEIGHT_DECAY=0.01,
              WARMUP_EPOCHS=1, USE_AMP=False, REPORT_TOP1000_COMPAT=True,
              ACTIVE_VISION_PROFILE='fixture', VISION_PROFILES={'fixture': {}},
              VAL_SUBSET_SEED=2026, BOOTSTRAP_SAMPLES=20, device=torch.device('cpu'))
    for i in [2, 7, 9, 12, 15, 17, 19, 24, 26, 27, 39, 74]:
        tree = ast.parse(cells[i])
        # These cells contain only definitions/constants; other cells also execute runs.
        if i not in [2, 9, 12, 15, 17, 19, 26, 27]:
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, f'notebook_cell_{i}', 'exec'), ns)
    return ns


class DummyTokenizer:
    def __call__(self, question, **kwargs):
        n = kwargs['max_length']
        return {'input_ids': torch.ones(1, n, dtype=torch.long),
                'attention_mask': torch.ones(1, n, dtype=torch.long)}


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.ns = definitions()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_every_code_cell_parses(self):
        notebook = json.loads(NOTEBOOK.read_text())
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                code = ''.join(cell['source'])
                ast.parse('\n'.join(line for line in code.splitlines() if not line.startswith(('!', '%'))), filename=f'cell_{i}')
                self.assertEqual(cell.get('outputs'), [])

    def test_official_normalization_and_leave_one_out_scores(self):
        cases = {'none': '0', 'twenty': 'twenty', 'couldve': "could've", '2.': '2',
                 '100,978': '100978', 'The TWO cats!': '2 cats', '2.50': '2.50',
                 'red/blue': 'red blue', 'a red, blue': 'red blue'}
        for text, expected in cases.items():
            self.assertEqual(self.ns['normalize_answer'](text), expected)
        for count, expected in enumerate([0, .3, .6, .9, 1, 1, 1, 1, 1, 1, 1]):
            self.assertAlmostEqual(self.ns['official_score'](count), expected)
        unanimous = [{'answer': 'two'} for _ in range(10)]
        self.assertEqual(self.ns['official_answer_counts'](unanimous), {'two': 10})
        unanimous[0]['answer'] = '2'
        self.assertEqual(self.ns['official_answer_counts'](unanimous), {'2': 10})

    def test_parameter_counts_flow_masks_gradients_and_attention_switch(self):
        n = self.ns
        for model_type in ['AsymmetricCrossModalFusion', 'ParameterMatchedSymmetricCrossModalFusion']:
            model = n[model_type](512, 8, dropout=0)
            self.assertEqual(n['count_parameters'](model), 6_306_816)
            del model
        n.update(EMBED_DIM=512, NUM_HEADS=8)
        n['audit_active_architectures']()
        model = n['AsymmetricCrossModalFusion'](16, 4, dropout=0)
        image, text = torch.randn(2, 5, 16), torch.randn(2, 3, 16)
        mask = torch.tensor([[False, False, True], [False, False, True]])
        seen = []
        handle = model.cross_attn_txt_to_img.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs['key_value']), with_kwargs=True)
        fast = model(image, text, mask)
        detailed = model(image, text, mask, return_attention=True)
        handle.remove()
        self.assertIs(seen[0], image)
        self.assertIsNone(fast[2])
        torch.testing.assert_close(fast[0], detailed[0])
        torch.testing.assert_close(fast[1], detailed[1])
        changed = text.clone(); changed[:, -1] = 10000
        torch.testing.assert_close(model(image, changed, mask)[0], fast[0])

    def test_full_model_parameter_parity_without_downloading_weights(self):
        n = self.ns
        clip = CLIPVisionConfig(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                                num_attention_heads=12, image_size=224, patch_size=32)
        roberta = RobertaConfig(vocab_size=50265, hidden_size=768, intermediate_size=3072,
                               num_hidden_layers=12, num_attention_heads=12,
                               max_position_embeddings=514, type_vocab_size=1, pad_token_id=1)
        n.update(IMAGE_SIZE=224, IMAGE_PATCH_SIZE=32, IMAGE_SEQ_LEN=50)
        with torch.device('meta'), \
                patch.object(CLIPVisionModel, 'from_pretrained', side_effect=lambda *a, **k: CLIPVisionModel(clip)), \
                patch.object(RobertaModel, 'from_pretrained', side_effect=lambda *a, **k: RobertaModel(roberta)):
            for name in ['AsymmetricVQAModelE2E', 'ParameterMatchedSymmetricVQAModelE2E']:
                model = n[name](3000, 512, 8, freeze_encoders=False)
                self.assertEqual(n['count_parameters'](model), 221_259_704)
                self.assertEqual(n['count_parameters'](model, True), 221_259_704)

    def test_dataset_keeps_oov_validation_and_reproducible_crops(self):
        n = self.ns
        questions = [{'question_id': i, 'image_id': i, 'question': 'What is shown?'} for i in range(3)]
        answers = [['yes'] * 10, ['red'] * 3 + ['blue'] * 7, ['unseen'] * 10]
        annotations = [{'question_id': i, 'answers': [{'answer': a, 'answer_id': j} for j, a in enumerate(row)],
                        'multiple_choice_answer': row[-1], 'answer_type': 'other'} for i, row in enumerate(answers)]
        (self.root / 'q.json').write_text(json.dumps({'questions': questions}))
        (self.root / 'a.json').write_text(json.dumps({'annotations': annotations}))
        with n['h5py'].File(self.root / 'images.h5', 'w') as f:
            f['image_ids'] = np.arange(3)
            f['images'] = np.random.default_rng(0).integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
        kwargs = dict(questions_file=self.root / 'q.json', annotations_file=self.root / 'a.json',
                      h5_path=self.root / 'images.h5', answer_to_idx={'yes': 0, 'red': 1, 'blue': 2})
        with patch.object(n['RobertaTokenizer'], 'from_pretrained', return_value=DummyTokenizer()):
            val = n['VQADataset'](**kwargs)
            train = n['VQADataset'](**kwargs, training=True, transform=n['get_image_transform']('train'))
        self.assertEqual(len(val), 3)
        self.assertEqual(len(train), 2)
        self.assertEqual(val.samples[2]['answer_target'].sum().item(), 0)
        self.assertAlmostEqual(val.samples[1]['answer_target'][1].item(), .9, places=6)
        self.assertEqual(train.samples[1]['answer_target'][1].item(), 1)
        train.set_epoch(42, 2); a = train[0][0]
        torch.rand(77); b = train[0][0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        train.set_epoch(42, 3)
        self.assertFalse(torch.equal(a, train[0][0]))
        class AlwaysYes(torch.nn.Module):
            def forward(self, image, ids, mask):
                return torch.tensor([[10., 0., 0.]]).expand(len(image), -1), {}
        loader = n['DataLoader'](val, batch_size=2, shuffle=False)
        metrics, predictions = n['evaluate'](AlwaysYes(), loader, torch.nn.BCEWithLogitsLoss(), False,
                                           collect_predictions=True)
        self.assertAlmostEqual(metrics['val_vqa_acc'], 100 / 3)
        np.testing.assert_array_equal(predictions['question_id'], [0, 1, 2])
        if train._h5f is not None: train._h5f.close()
        if val._h5f is not None: val._h5f.close()

    def tiny_training(self):
        n = self.ns
        dataset = torch.utils.data.TensorDataset(torch.randn(6, 5, 16), torch.randn(6, 4, 16),
                                                 torch.ones(6, 4, dtype=torch.long), torch.eye(3).repeat(2, 1))
        dataset.samples = [{'question_id': i, 'image_id': i // 2} for i in range(6)]
        loader = n['DataLoader'](dataset, batch_size=2, shuffle=False)
        n.update(train_ds=dataset, train_loader=loader, val_ds=dataset, val_loader=loader,
                 val_subset_loader=loader, val_subset_indices=list(range(6)),
                 answer_priors=torch.tensor([.2, .3, .4]), METRICS_DIR=self.root,
                 CHECKPOINT_DIR=self.root, RUN_TAG='fixture', CONFIG_SHA256='fixture_hash',
                 EXPERIMENT_CONFIG={}, METRIC_VERSION='official_fixture')
        n['EXPECTED_FUSION_PARAMS'] = n['count_parameters'](n['AsymmetricCrossModalFusion'](16, 4))
        return n

    def test_paired_initialization_epoch_order_and_resume(self):
        n = self.tiny_training()
        # Exercise scheduler restoration too, without allocating real encoders.
        original_builder = n['build_optimizer_and_scheduler']
        def scheduled_optimizer(model):
            optimizer, _, groups = original_builder(model)
            scheduler = n['get_cosine_schedule_with_warmup'](
                optimizer, 0, n['EPOCHS'] * len(n['train_loader']))
            return optimizer, scheduler, groups
        n['build_optimizer_and_scheduler'] = scheduled_optimizer
        sym = n['make_paired_model']('symmetric_matched', 7)
        asym = n['make_paired_model']('asymmetric', 7)
        self.assertEqual(sym.initialization_hashes, asym.initialization_hashes)
        torch.testing.assert_close(sym.classifier[-1].bias.sigmoid(), n['answer_priors'])
        order = torch.cat([b[0] for b in n['make_epoch_train_loader'](7, 2)])
        torch.rand(100)
        torch.testing.assert_close(order, torch.cat([b[0] for b in n['make_epoch_train_loader'](7, 2)]))
        saved = {}
        original_save = n['save_checkpoint']
        def save_and_capture(payload, path):
            original_save(payload, path)
            if payload.get('epoch') == 1:
                saved[Path(path).name] = copy.deepcopy(payload)
        n['save_checkpoint'] = save_and_capture
        uninterrupted, _ = n['run_training']('asymmetric', seed=7)
        expected = copy.deepcopy(uninterrupted.state_dict())
        name = n['run_name_for'](7, 'asymmetric')
        n['save_checkpoint'] = original_save
        for filename, payload in saved.items(): original_save(payload, self.root / filename)
        (self.root / f'{name}_history.json').write_text(json.dumps(saved[f'{name}_latest.pt']['history']))
        resumed, history = n['run_training']('asymmetric', seed=7)
        self.assertEqual(len(history), 3)
        for key, value in resumed.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            n['load_run_checkpoint'](self.root / f'{name}_latest.pt', 'symmetric_matched', 7)

    def test_scheduler_does_not_advance_on_a_skipped_amp_update(self):
        n = self.tiny_training()
        model = n['make_paired_model']('asymmetric', 7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        class SkipFirstUpdate:
            def __init__(self): self.current_scale = 2; self.calls = 0
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.current_scale
            def step(self, optimizer):
                if self.calls: optimizer.step()
            def update(self): self.current_scale = 1; self.calls += 1
        metrics = n['train_one_epoch'](model, n['train_loader'], torch.nn.BCEWithLogitsLoss(),
                                       optimizer, SkipFirstUpdate(), False, scheduler=scheduler)
        self.assertEqual(metrics['skipped_optimizer_steps'], 1)
        self.assertEqual(metrics['optimizer_steps'], 2)
        self.assertEqual(scheduler.last_epoch, 2)

    def test_export_coverage_selection_and_paired_uncertainty(self):
        n = self.ns
        valid = [{'question_id': i, 'answer': 'yes'} for i in range(3)]
        questions = [{'question_id': i} for i in range(3)]
        n['validate_test_predictions'](valid, questions)
        for invalid in [valid[:-1], valid + valid[:1], [{'question_id': 0, 'answer': ''}]]:
            with self.assertRaises(ValueError): n['validate_test_predictions'](invalid, questions)
        values = {'question_id': np.arange(4), 'image_id': np.array([1, 1, 2, 2]), 'score': np.zeros(4)}
        better = dict(values, score=np.ones(4))
        self.assertEqual(n['paired_image_bootstrap'](values, better), [100., 100.])
        runs = [{'models': {v: {'seed': seed, 'best_epoch': 2, 'best_subset_vqa_acc': 60.0}
                           for v in n['ACTIVE_MODEL_TYPES']}} for seed in [42, 7]]
        self.assertTrue(all(r['seed'] == 7 for r in n['select_campaign_checkpoints'](runs).values()))

    def test_campaign_report_and_export_on_synthetic_features(self):
        n = self.tiny_training()
        n.update(GPU_BUDGET_HOURS=12, EXPORT_RESERVE_HOURS=1.5,
                 preflight_report={'elapsed_s': 0, 'models': {
                     v: {'epoch_seconds': .01, 'full_val_seconds': .01} for v in n['ACTIVE_MODEL_TYPES']}})
        runs = n['run_campaign']()
        self.assertEqual([r['seed'] for r in runs], [7, 42, 123])
        n['campaign_runs'] = runs
        n['selected_runs'] = n['select_campaign_checkpoints'](runs)
        # The report cell uses persisted per-question predictions from each seed.
        notebook = json.loads(NOTEBOOK.read_text())
        exec(''.join(notebook['cells'][39]['source']), n)
        self.assertEqual(n['final_training_report']['seeds_completed'], [7, 42, 123])
        selected, _ = n['load_selected_model']('asymmetric')
        data = n['train_ds'].tensors
        test_loader = n['DataLoader'](torch.utils.data.TensorDataset(*data[:3], torch.arange(6)), batch_size=2)
        preds = n['predict_test'](selected, test_loader, {0: 'yes', 1: 'no', 2: 'red'})
        n['validate_test_predictions'](preds, [{'question_id': i} for i in range(6)])

    def test_budget_guard_does_not_start_an_unaffordable_pair(self):
        n = self.tiny_training()
        n.update(GPU_BUDGET_HOURS=1, EXPORT_RESERVE_HOURS=1.5,
                 preflight_report={'elapsed_s': 0, 'models': {
                     v: {'epoch_seconds': 1, 'full_val_seconds': 1} for v in n['ACTIVE_MODEL_TYPES']}})
        with self.assertRaisesRegex(RuntimeError, 'No complete paired run'):
            n['run_campaign']()


if __name__ == '__main__':
    unittest.main()
