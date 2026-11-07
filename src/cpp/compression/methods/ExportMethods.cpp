//
// Created by Marcos Perez on 25/7/26.
//
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
//#include <torch/csrc/distributed/c10d/python_comm.hpp>
//#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>
//#include <torch/extension.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"   // support for loading levels from the environment variable
#include "spdlog/fmt/ostr.h"  // support for user defined types
import PowerSGD;
import CompressionFilter;
import Metrics;
import Network;
import NoCompressionMethod;
import QSGD;
import RMQ;
import DAQ;

using compression::methods::DoublePowerSGD;
using compression::methods::DoubleQSGD;
using compression::methods::FloatQSGD;
using compression::metrics::Metrics;
using compression::metrics::NumberTracking;
using compression::metrics::VectorMetric;
namespace py = pybind11;


PYBIND11_MODULE(compression_python, m) {
    //py::class_<CompressionFilter>(m, "CompressionFilter");
    //
    //py::class_<QSGDFilter, CompressionFilter>(m, "QSGDFilter")
    //    .def(py::init<int>())
    //    .def("compress", &QSGDFilter::compress)
    //    .def("decompress", &QSGDFilter::decompress);
    py::class_<VectorMetric>(m, "VectorMetric")
            .def_property_readonly("name", &VectorMetric::name)
            .def_property_readonly(
                "registered_chunk_count",
                &VectorMetric::registered_chunk_count)
            .def_property_readonly("norm_0", &VectorMetric::norm_0)
            .def_property_readonly("norm_1", &VectorMetric::norm_1)
            .def_property_readonly("norm_2", &VectorMetric::norm_2)
            .def_property_readonly("norm_inf", &VectorMetric::norm_inf)
            .def_property_readonly("bytes", &VectorMetric::bytes)
            .def_property_readonly(
                "total_elements",
                &VectorMetric::total_elements)
            .def("reset", &VectorMetric::reset);

    py::class_<NumberTracking<double> >(m, "NumberMetric")
            .def_property_readonly("name", &NumberTracking<double>::name)
            .def_property_readonly("value", &NumberTracking<double>::value)
            .def_property_readonly(
                "running_average",
                &NumberTracking<double>::running_average)
            .def_property_readonly(
                "aggregated_number",
                &NumberTracking<double>::aggregated_number)
            .def_property_readonly(
                "min",
                &NumberTracking<double>::min)
            .def_property_readonly(
                "max",
                &NumberTracking<double>::max)
            .def_property_readonly(
                "total_calls",
                &NumberTracking<double>::total_calls)
            .def_property_readonly("avg", &NumberTracking<double>::avg)
            .def(
                "reset",
                &NumberTracking<double>::reset,
                py::arg("full_reset") = false);

    m.def("has_vector_metric",
          [](const std::string &name) {
              return Metrics::metrics().has_vector_metric(name);
          });
    m.def("has_number_metric",
          [](const std::string &name) {
              return Metrics::metrics().has_number_metric(name);
          });
    m.def(
        "vector_metric",
        [](const std::string &name) -> VectorMetric & {
            return Metrics::metrics().vector(name);
        },
        py::return_value_policy::reference);

    m.def(
        "number_metric",
        [](const std::string &name) -> NumberTracking<double> & {
            return Metrics::metrics().number(name);
        },
        py::return_value_policy::reference);

    m.def("reset_metrics", [](bool full_reset) {
        Metrics::metrics().reset(full_reset);
    }, py::arg("full_reset") = false);

    py::class_<compression::base::CompressionFilterCtx>(m, "CompressionFilterCtx")
            .def(py::init<int>());

    py::class_<DoublePowerSGD>(m, "PowerSGD")
            .def(py::init<int>())
            .def("run", &DoublePowerSGD::doMethod);

    py::class_<FloatQSGD>(m, "QSGD")
            .def(py::init<int, int, int, bool>())
            .def("run", &FloatQSGD::doMethod);

    py::class_<compression::methods::NoCompressionMethod>(m, "NoCompression")
            .def(py::init<>())
            .def("run", &compression::methods::NoCompressionMethod::doMethod);

    py::class_<compression::methods::RMQMethod>(m, "RMQ")
            .def(py::init<double, int>())
            .def("run", &compression::methods::RMQMethod::doMethod);

    py::class_<compression::methods::DAQMethod>(m, "DAQ")
            .def(py::init<int>())
            .def("run", &compression::methods::DAQMethod::doMethod);

    m.def("sent_bytes", []()-> uint64_t {
        return Network::instance().get_sent_bytes();
    });
    m.def("recv_bytes", []()-> uint64_t {
        return Network::instance().get_recv_bytes();
    });

    m.def("reset_network_stats", []() {
        Network::instance().reset_network_stats();
    });

    m.def("initialize", [](py::object pg) {
        py::object boxed_pg = pg.attr("boxed")();
        auto process_group_type = torch::getCustomClass(
            "__torch__.torch.classes.c10d.ProcessGroup"
        );
        auto process_group = torch::jit::toIValue(
            std::move(boxed_pg),
            process_group_type
        ).toCustomClass<c10d::ProcessGroup>();

        spdlog::cfg::load_env_levels();
        Network::instance().initialize(
            std::move(process_group),
            Topology::RING
        );
    });
}
