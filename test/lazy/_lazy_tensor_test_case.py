from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch
import torch
import os
import random
from abc import abstractmethod
from gpytorch.utils import approx_equal


class LazyTensorTestCase(object):
    should_test_sample = False

    @abstractmethod
    def create_lazy_tensor(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_lazy_tensor(self):
        raise NotImplementedError()

    def setUp(self):
        if hasattr(self.__class__, "seed"):
            seed = self.__class__.seed
            if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
                self.rng_state = torch.get_rng_state()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1))
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        assert(approx_equal(res, actual))

        grad = torch.randn_like(test_vector)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                assert(approx_equal(arg.grad, arg_copy.grad))

    def test_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1), 5)
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        assert(approx_equal(res, actual))

        grad = torch.randn_like(test_vector)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                assert(approx_equal(arg.grad, arg_copy.grad))

    def test_inv_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1))
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector)
        assert(approx_equal(res, actual))

        grad = torch.randn_like(test_vector)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                assert(approx_equal(arg.grad, arg_copy.grad))

    def test_inv_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1), 5)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector)
        assert(approx_equal(res, actual))

        grad = torch.randn_like(test_vector)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                assert(approx_equal(arg.grad, arg_copy.grad))

    def test_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        self.assertTrue(approx_equal(lazy_tensor.diag(), evaluated.diag()))

    def test_evaluate(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertTrue(approx_equal(lazy_tensor.evaluate(), evaluated))

    def test_inv_quad_log_det(self):
        # Forward
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        vecs = torch.randn(lazy_tensor.size(1), 3, requires_grad=True)
        vecs_copy = vecs.clone()

        res_inv_quad, res_log_det = lazy_tensor.inv_quad_log_det(inv_quad_rhs=vecs, log_det=True)
        res = res_inv_quad + res_log_det

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum()
        actual = actual_inv_quad + torch.logdet(evaluated)

        diff = (res - actual).abs() / actual.abs().clamp(1, 1e10)
        self.assertLess(diff.item(), 5e-2)

    def test_sample(self):
        if self.__class__.should_test_sample:
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            samples = lazy_tensor.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertLess(((sample_covar - evaluated).abs() / evaluated.abs().clamp(1, 1e5)).max().item(), 3e-1)

    def test_getitem(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        self.assertTrue(approx_equal(lazy_tensor[2], evaluated[2]))
        self.assertTrue(approx_equal(lazy_tensor[0:2].evaluate(), evaluated[0:2]))
        self.assertTrue(approx_equal(lazy_tensor[:, 2:3].evaluate(), evaluated[:, 2:3]))
        self.assertTrue(approx_equal(lazy_tensor[:, 0:2].evaluate(), evaluated[:, 0:2]))
        self.assertTrue(approx_equal(lazy_tensor[:1, :2].evaluate(), evaluated[:1, :2]))
        self.assertTrue(approx_equal(lazy_tensor[1, :2], evaluated[1, :2]))
        self.assertTrue(approx_equal(lazy_tensor[:1, 2], evaluated[:1, 2]))

    def test_getitem_tensor_index(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))


class BatchLazyTensorTestCase(object):
    should_test_sample = False

    @abstractmethod
    def create_lazy_tensor(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_lazy_tensor(self):
        raise NotImplementedError()

    def setUp(self):
        if hasattr(self.__class__, "seed"):
            seed = self.__class__.seed
            if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
                self.rng_state = torch.get_rng_state()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(0), lazy_tensor.size(-1), 5)
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        assert(approx_equal(res, actual))

        grad = torch.randn_like(test_vector)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                assert(approx_equal(arg.grad, arg_copy.grad))

    def test_inv_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(0), lazy_tensor.size(-1), 5)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = torch.cat([
            evaluated[i].inverse().matmul(test_vector[i]).unsqueeze(0) for i in range(lazy_tensor.size(0))
        ])
        assert(approx_equal(res, actual))

        grad = torch.randn_like(test_vector)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                assert(approx_equal(arg.grad, arg_copy.grad))

    def test_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        actual = torch.stack([evaluated[i].diag() for i in range(evaluated.size(0))])
        self.assertTrue(approx_equal(lazy_tensor.diag(), actual))

    def test_evaluate(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertTrue(approx_equal(lazy_tensor.evaluate(), evaluated))

    def test_inv_quad_log_det(self):
        # Forward
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        vecs = torch.randn(lazy_tensor.size(0), lazy_tensor.size(1), 3, requires_grad=True)
        vecs_copy = vecs.clone()

        res_inv_quad, res_log_det = lazy_tensor.inv_quad_log_det(inv_quad_rhs=vecs, log_det=True)
        res = res_inv_quad + res_log_det

        actual_inv_quad = torch.cat([
            evaluated[i].inverse().matmul(vecs_copy[i]).mul(vecs_copy[i]).sum().unsqueeze(0)
            for i in range(lazy_tensor.size(0))
        ])
        actual = actual_inv_quad + torch.cat([
            torch.logdet(evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.size(0))
        ])

        diffs = (res - actual).abs() / actual.abs().clamp(1, 1e10)
        for i in range(lazy_tensor.size(0)):
            self.assertLess(diffs[i].item(), 5e-2)

    def test_sample(self):
        if self.__class__.should_test_sample:
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            samples = lazy_tensor.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertLess(((sample_covar - evaluated).abs() / evaluated.abs().clamp(1, 1e5)).max().item(), 3e-1)

    def test_getitem(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        self.assertTrue(approx_equal(lazy_tensor[1].evaluate(), evaluated[1]))
        self.assertTrue(approx_equal(lazy_tensor[0:2].evaluate(), evaluated[0:2]))
        self.assertTrue(approx_equal(lazy_tensor[:, 2:3].evaluate(), evaluated[:, 2:3]))
        self.assertTrue(approx_equal(lazy_tensor[:, 0:2].evaluate(), evaluated[:, 0:2]))
        self.assertTrue(approx_equal(lazy_tensor[1, :1, :2].evaluate(), evaluated[1, :1, :2]))
        self.assertTrue(approx_equal(lazy_tensor[1, 1, :2], evaluated[1, 1, :2]))
        self.assertTrue(approx_equal(lazy_tensor[1, :1, 2], evaluated[1, :1, 2]))

    def test_getitem_tensor_index(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        print(evaluated)
        print(lazy_tensor.evaluate())
        print(lazy_tensor[index])
        print(evaluated[index])
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
