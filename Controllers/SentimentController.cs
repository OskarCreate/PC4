using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using PC4.Models;

namespace PC4.Controllers
{
    public class SentimentController : Controller
    {
        private readonly PredictionEngine<SentimentData, SentimentPrediction> _predictionEngine;

        public SentimentController()
        {
            var mlContext = new MLContext();
            DataViewSchema modelSchema;
            var model = mlContext.Model.Load("sentiment_model.zip", out modelSchema);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Predict(SentimentData input)
        {
            if (ModelState.IsValid)
            {
                var prediction = _predictionEngine.Predict(input);
                ViewBag.Result = prediction.Prediction ? "POSITIVO" : "NEGATIVO";
                ViewBag.Probability = prediction.Probability.ToString("P2");
            }
            return View("Index");
        }
    }
}
