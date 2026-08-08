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
import Network;
import NoCompressionMethod;


using compression::methods::DoublePowerSGD;
namespace py = pybind11;


PYBIND11_MODULE(compression_python, m) {
    //py::class_<CompressionFilter>(m, "CompressionFilter");
    //
    //py::class_<QSGDFilter, CompressionFilter>(m, "QSGDFilter")
    //    .def(py::init<int>())
    //    .def("compress", &QSGDFilter::compress)
    //    .def("decompress", &QSGDFilter::decompress);
    py::class_<compression::base::CompressionFilterCtx>(m, "CompressionFilterCtx")
            .def(py::init<int>());

    py::class_<DoublePowerSGD>(m, "PowerSGD")
            .def(py::init<int>())
            .def("run", &DoublePowerSGD::doMethod);

    py::class_<compression::methods::NoCompressionMethod>(m, "NoCompression")
            .def(py::init<>())
            .def("run", &compression::methods::NoCompressionMethod::doMethod);

    m.def("get_bytes", []()-> uint64_t {
        return Network::instance().get_bytes();
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
