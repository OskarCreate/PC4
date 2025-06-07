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
    public class RecommendationController : Controller
    {
        private readonly PredictionEngine<RatingData, RatingPrediction> _predictionEngine;
        private readonly string[] _productIds = { "P1", "P2", "P3", "P4", "P5" };

        public RecommendationController()
        {
            var mlContext = new MLContext();
            DataViewSchema modelSchema;
            var model = mlContext.Model.Load("recommendation_model.zip", out modelSchema);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<RatingData, RatingPrediction>(model);
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Recommend(string userId)
        {
            var topProducts = _productIds
                .Select(pid => new
                {
                    ProductId = pid,
                    Score = _predictionEngine.Predict(new RatingData { UserId = userId, ProductId = pid }).Score
                })
                .OrderByDescending(p => p.Score)
                .Take(3)
                .ToList();

            ViewBag.Recommendations = topProducts;
            ViewBag.UserId = userId;
            return View("Index");
        }
    }
}
