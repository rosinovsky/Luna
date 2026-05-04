#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "clinker_forecast.h"
#include <cstring>

namespace py = pybind11;

// Python wrapper for CF_Model
class PyModel {
    CF_Model* model_;
public:
    PyModel(const std::string& path) {
        CF_Status status;
        model_ = cf_load_model(path.c_str(), &status);
        if (!model_) {
            throw std::runtime_error("Failed to load model: " + std::to_string(status));
        }
    }

    ~PyModel() {
        if (model_) cf_free_model(model_);
    }

    py::dict info() const {
        CF_ModelInfo info = cf_get_model_info(model_);
        py::dict d;
        d["version"] = info.version;
        d["created_at"] = info.created_at;
        d["d_model"] = info.d_model;
        d["n_heads"] = info.n_heads;
        d["n_layers"] = info.n_layers;
        d["history_len"] = info.history_len;
        d["n_features"] = info.n_features;
        d["n_outputs"] = info.n_outputs;
        d["n_quantiles"] = info.n_quantiles;
        d["total_params"] = info.total_params;
        d["model_size_mb"] = info.model_size_bytes / (1024.0 * 1024.0);
        return d;
    }

    py::dict predict(py::dict tech_dict, py::array_t<float> history, int horizon = 24) {
        // Parse tech params
        CF_TechnologyParams tech;
        memset(&tech, 0, sizeof(tech));

        if (tech_dict.contains("temperatures")) {
            auto temps = tech_dict["temperatures"].cast<py::list>();
            tech.n_zones = static_cast<int>(temps.size());
            for (int i = 0; i < tech.n_zones && i < CF_MAX_ZONES; i++) {
                tech.temperatures[i] = temps[i].cast<float>();
            }
        }
        if (tech_dict.contains("flows")) {
            auto flows = tech_dict["flows"].cast<py::list>();
            tech.n_flows = static_cast<int>(flows.size());
            for (int i = 0; i < tech.n_flows && i < CF_MAX_FLOWS; i++) {
                tech.flows[i] = flows[i].cast<float>();
            }
        }
        if (tech_dict.contains("kiln_speed")) {
            tech.kiln_speed = tech_dict["kiln_speed"].cast<float>();
        }
        if (tech_dict.contains("chemistry")) {
            auto chem = tech_dict["chemistry"].cast<py::list>();
            tech.n_chemistry = static_cast<int>(chem.size());
            for (int i = 0; i < tech.n_chemistry && i < CF_MAX_CHEMISTRY; i++) {
                tech.chemistry[i] = chem[i].cast<float>();
            }
        }

        // Parse history
        CF_TimeSeries hist;
        memset(&hist, 0, sizeof(hist));
        auto buf = history.request();
        if (buf.ndim == 2) {
            hist.history_len = static_cast<int>(buf.shape[0]);
            hist.n_features = static_cast<int>(buf.shape[1]);
            float* ptr = static_cast<float*>(buf.ptr);
            memcpy(hist.data, ptr, 
                   std::min(hist.history_len * hist.n_features, 
                           CF_MAX_HISTORY_LEN * CF_MAX_FEATURES) * sizeof(float));
        }

        // Predict
        CF_Result* result = cf_predict_single(model_, &tech, &hist, 
                                               time(nullptr), horizon);
        if (!result) {
            throw std::runtime_error("Prediction failed");
        }

        // Convert to Python dict
        py::dict out;
        out["inference_time_ms"] = result->inference_time_ms;
        out["status"] = (result->status == CF_OK) ? "ok" : "error";

        if (result->batch_size > 0) {
            const CF_Prediction& p = result->predictions[0];

            py::dict pred;
            pred["horizon_hours"] = p.horizon_hours;

            py::dict c3s;
            c3s["q05"] = p.c3s[0];
            c3s["q50"] = p.c3s[1];
            c3s["q95"] = p.c3s[2];
            pred["c3s"] = c3s;

            py::dict c2s;
            c2s["q05"] = p.c2s[0];
            c2s["q50"] = p.c2s[1];
            c2s["q95"] = p.c2s[2];
            pred["c2s"] = c2s;

            py::dict cao;
            cao["q05"] = p.free_cao[0];
            cao["q50"] = p.free_cao[1];
            cao["q95"] = p.free_cao[2];
            pred["free_cao"] = cao;

            py::dict lit;
            lit["q05"] = p.liter_weight[0];
            lit["q50"] = p.liter_weight[1];
            lit["q95"] = p.liter_weight[2];
            pred["liter_weight"] = lit;

            pred["trend"] = (p.trend == -1) ? "decreasing" :
                           (p.trend == 1) ? "increasing" : "stable";
            pred["trend_confidence"] = p.trend_confidence;
            pred["physics_valid"] = p.physics_valid;
            if (!p.physics_valid) {
                pred["physics_error"] = p.physics_error;
            }

            // Feature importance as numpy array
            py::array_t<float> feat_imp({CF_MAX_FEATURES});
            float* feat_ptr = static_cast<float*>(feat_imp.request().ptr);
            memcpy(feat_ptr, p.feature_importance, CF_MAX_FEATURES * sizeof(float));
            pred["feature_importance"] = feat_imp;

            out["prediction"] = pred;
        }

        cf_free_result(result);
        return out;
    }
};

PYBIND11_MODULE(clinker_forecast, m) {
    m.doc() = "ClinkerForecast: Neural forecasting engine for cement clinker quality";

    py::class_<PyModel>(m, "Model")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("info", &PyModel::info)
        .def("predict", &PyModel::predict, 
             py::arg("tech"), py::arg("history"), py::arg("horizon") = 24)
        .def("__repr__", [](const PyModel& m) {
            auto info = m.info();
            return "<ClinkerForecast.Model v" + info["version"].cast<std::string>() + ">";
        });

    m.def("version", []() {
        return cf_version();
    });
}