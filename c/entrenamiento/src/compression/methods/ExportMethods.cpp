//
// Created by Marcos Perez on 25/7/26.
//
#include <pybind11/pybind11.h>
import PowerSGD;

export module ExportMethods;

import PowerSGD;

using compression::methods::DoublePowerSGD;
namespace py = pybind11;

PYBIND11_MODULE(compression, m) {
    //py::class_<CompressionFilter>(m, "CompressionFilter");
    //
    //py::class_<QSGDFilter, CompressionFilter>(m, "QSGDFilter")
    //    .def(py::init<int>())
    //    .def("compress", &QSGDFilter::compress)
    //    .def("decompress", &QSGDFilter::decompress);

    py::class_<DoublePowerSGD>(m, "PowerSGD")
            .def(py::init<int>())
            .def("run", &DoublePowerSGD::doMethod);
}
