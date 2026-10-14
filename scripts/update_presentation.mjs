import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "file:///C:/Users/skyda/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const workspaceDir = path.resolve(".");
process.env.RUNTIME_NODE_MODULES = "C:/Users/skyda/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const sourcePath = path.join(workspaceDir, "reports/Презентация_к_защите_SBER_проверенная.pptx");
const outputPath = path.join(workspaceDir, "reports/Презентация_к_защите_SBER_итог.pptx");
const buildDir = path.join(workspaceDir, ".codex-pptx-build");
const candidatePath = path.join(buildDir, "candidate.pptx");
const slidesPreviewDir = path.join(buildDir, "slides");
await fs.mkdir(slidesPreviewDir, { recursive: true });

const comparison = JSON.parse(await fs.readFile(path.join(workspaceDir, "reports/model_comparison.json"), "utf8"));
const featureAnalysis = JSON.parse(await fs.readFile(path.join(workspaceDir, "reports/feature_analysis.json"), "utf8"));
const monitoring = JSON.parse(await fs.readFile(path.join(workspaceDir, "reports/monitoring.json"), "utf8"));
const predictionText = await fs.readFile(path.join(workspaceDir, "reports/test_predictions.csv"), "utf8");
const predictionLines = predictionText.trim().split(/\r?\n/);
const predictionHeaders = predictionLines[0].split(",");
const predictionRows = predictionLines.slice(1).map(line => {
  const values = line.split(",");
  return Object.fromEntries(predictionHeaders.map((header, index) => [header, values[index]]));
});

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const updateText = (id, value) => { presentation.resolve(id).text = value; };

// Cyclical calendar encoding and the limited RSS archive.
updateText("sh/3ihk3et8", "Цена и объём\n\nДоходности и лаги 1, 2 и 5 сессий\n\nСредние цены за 5 и 20 сессий\n\nВолатильность за 5 сессий\n\nКалендарь: sin/cos циклы\n\nТекст RSS обработан, но покрытие слишком мало для обучения");

// Validation chart compares each model after its feature set has been chosen on validation.
const validationSlide = presentation.slides.items[5];
validationSlide.charts.deleteById("6");
const validationOrder = ["naive_last_close", "gradient_boosting", "random_forest", "linear_regression"];
validationSlide.charts.add("bar", {
  position: { left: 92, top: 177, width: 790, height: 400 },
  categories: ["Последнее закрытие", "Градиентный бустинг", "Случайный лес", "Линейная регрессия"],
  series: [{ name: "MAE, руб.", values: validationOrder.map(name => Number(comparison.validation_metrics[name].mae_rub.toFixed(2))), fill: "#4472C4" }],
  hasLegend: false, barOptions: { direction: "bar", grouping: "clustered" },
  xAxis: { textStyle: { fontSize: 12 } }, yAxis: { title: "MAE, руб.", min: 0, max: 4, majorUnit: 1 },
});
updateText("sh/pc76hkr2", "Линейная регрессия выбрана на validation");
updateText("sh/yhg7epsj", "MAE, руб. · 16.10.2024–14.10.2025 · 252 сессии · набор признаков выбирался по validation");
updateText("sh/7m98ru9g", "3,29 ₽\nMAE линейной модели");
updateText("sh/oryp8fah", "Выбран набор price_levels_only из пяти ценовых признаков. Преимущество над базовым прогнозом небольшое: 3,29 против 3,31 руб. Тест не использовался при выборе.");
presentation.resolve("nt/x8f69ofe").text = "По validation MAE выбрана линейная регрессия с пятью ценовыми признаками: 3,2853 руб. Наивный прогноз: 3,3084 руб.; случайный лес: 3,3917 руб.; градиентный бустинг: 3,3880 руб. Одинаковая схема и временные границы; тест не участвовал в подборе.";

// Actual values, predictions, and metrics on the untouched test period.
const testSlide = presentation.slides.items[6];
testSlide.charts.deleteById("6");
const recentPredictions = predictionRows.slice(-60);
testSlide.charts.add("line", {
  position: { left: 80, top: 165, width: 1120, height: 365 },
  categories: recentPredictions.map(row => row.target_date),
  series: [
    { name: "Факт", values: recentPredictions.map(row => Number(Number(row.actual_close).toFixed(2))), line: { fill: "#4472C4", width: 2 } },
    { name: "Прогноз", values: recentPredictions.map(row => Number(Number(row.predicted_close).toFixed(2))), line: { fill: "#ED7D31", width: 2 } },
  ], hasLegend: true, lineOptions: { smooth: false },
  yAxis: { min: 250, max: 300, majorUnit: 10, numberFormatCode: "0,00" },
});
updateText("sh/l0vuh0rm", `MAE  ${comparison.untouched_test_metrics.mae_rub.toFixed(2).replace(".", ",")} ₽`);
updateText("sh/eh0ba1sr", `RMSE  ${comparison.untouched_test_metrics.rmse_rub.toFixed(2).replace(".", ",")} ₽`);
updateText("sh/fi9c369c", `MAPE  ${comparison.untouched_test_metrics.mape_percent.toFixed(2).replace(".", ",")}%`);
updateText("sh/baxkjqlk", `R²  ${comparison.untouched_test_metrics.r2.toFixed(3).replace(".", ",")}`);
presentation.resolve("nt/gnmp4jqx").text = `Отложенный тест: ${comparison.test_period.start}–${comparison.test_period.end}, ${comparison.test_period.rows} сессии. Выбрана линейная регрессия с пятью ценовыми признаками. MAE ${comparison.untouched_test_metrics.mae_rub.toFixed(4)} руб.; RMSE ${comparison.untouched_test_metrics.rmse_rub.toFixed(4)} руб.; MAPE ${comparison.untouched_test_metrics.mape_percent.toFixed(4)}%; R² ${comparison.untouched_test_metrics.r2.toFixed(4)}. Метрики цены не подтверждают торговую доходность.`;

// Keep the source coverage limitation explicit.
updateText("sh/98rehwve", "RSS не покрывает историю SBER");
presentation.resolve("nt/fu1gfa1s").append("\nОткрытая страница раскрытия информации ПАО Сбербанк изучена как возможный источник: https://disclosure.1prime.ru/Portal/Default.aspx?emid=7707083893 . Полная историческая выгрузка не подключена; источник не использовался в моделях.");

// Monitoring values follow the selected model and the regenerated holdout predictions.
const monitoringSlide = presentation.slides.items[8];
monitoringSlide.charts.deleteById("6");
const months = Object.keys(monitoring.monthly_mae).sort();
monitoringSlide.charts.add("bar", {
  position: { left: 70, top: 170, width: 740, height: 420 },
  categories: months,
  series: [{ name: "MAE, руб.", values: months.map(month => Number(monitoring.monthly_mae[month].mae_rub.toFixed(2))), fill: "#70AD47" }],
  hasLegend: false, barOptions: { direction: "column", grouping: "clustered" },
});
updateText("sh/ofq5svm5", `Последние 20 сессий\nMAE ${monitoring.recent_window_mae_rub.toFixed(2).replace(".", ",")} ₽\n\nВалидация\nMAE ${monitoring.reference_validation_mae_rub.toFixed(2).replace(".", ",")} ₽\n\nОтношение ${monitoring.recent_to_validation_mae_ratio.toFixed(2).replace(".", ",")} · в пределах порога`);
presentation.resolve("nt/udsvah03").text = `Исторический holdout monitor. Последние 20 сессий MAE ${monitoring.recent_window_mae_rub.toFixed(4)} руб.; validation MAE ${monitoring.reference_validation_mae_rub.toFixed(4)} руб.; отношение ${monitoring.recent_to_validation_mae_ratio.toFixed(4)}. Пороговая метка не сработала; это не доказательство отсутствия дрейфа.`;

// Update the conclusion to match the model and test metrics actually produced.
updateText("sh/l036l83y", "Линейная регрессия выбрана после отбора признаков на валидации.\n\nТестовая MAE составила 2,43 руб.\n\nАрхив RSS не позволяет оценить влияние новостей SBER.\n\nДальше нужны сопоставимая история публикаций и эксплуатационный мониторинг входных данных.");
presentation.resolve("nt/m90b6t0r").text = "Источники и результаты: MOEX ISS — https://www.moex.com/a2920 ; Банк России RSS — https://www.cbr.ru/development/RSS/ . Метрики и анализ признаков рассчитаны по сохранённым данным на 10.10.2026. Раскрытия Сбербанка на 1Prime изучены, но не вошли в обучение из-за отсутствия автоматизированной исторической выгрузки.";

// Add the completed feature selection and interpretation component.
const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";
const addText = (name, text, position, style) => {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = style;
  shape.text.insets = 0;
  return shape;
};
addText("title", "Отбор и объяснение признаков", { left: 72, top: 48, width: 1136, height: 54 }, { fontSize: 42, typeface: "Arial", color: "#142B45", bold: true });
addText("subtitle", "Группы сравнивались на validation; test сохранён для итоговой оценки", { left: 74, top: 106, width: 1130, height: 42 }, { fontSize: 20, typeface: "Arial", color: "#5B6573" });
addText("selection", "Выбор набора признаков\n\nПроверены 5 наборов для каждого ML-класса.\n\nЛинейная регрессия: price_levels_only — MAE 3,29 ₽.\n\nСлучайный лес: price_levels_only — MAE 3,39 ₽.\n\nГрадиентный бустинг: price_levels_only — MAE 3,39 ₽.\n\nБазовый прогноз: MAE 3,31 ₽.", { left: 90, top: 185, width: 505, height: 400 }, { fontSize: 20, typeface: "Arial", color: "#142B45" });
addText("explanation", "Оценка влияния признаков\n\nВстроенная важность деревьев и permutation importance сопоставлены между собой.\n\nPermutation importance рассчитана на validation и отдельно на тесте после выбора модели.\n\nДля цены и доходности рассчитана частичная зависимость на данных validation.\n\nТекущий прогноз линейной модели локально объясняется отклонением признаков от медиан train.", { left: 660, top: 185, width: 530, height: 400 }, { fontSize: 20, typeface: "Arial", color: "#142B45" });
addText("limitation", "RSS Банка России покрывает 0,50% строк и не содержит полноценной истории новостей SBER; текстовые признаки исключены из обучения.", { left: 90, top: 596, width: 1085, height: 55 }, { fontSize: 18, typeface: "Arial", color: "#5B6573" });
addText("footer", "ИАД · SBER · 2026", { left: 72, top: 678, width: 900, height: 22 }, { fontSize: 13, typeface: "Arial", color: "#5B6573" });
addText("page", "11", { left: 1150, top: 678, width: 58, height: 22 }, { fontSize: 13, typeface: "Arial", color: "#5B6573", alignment: "right" });
slide.speakerNotes.text = [
  "Показатели рассчитаны из reports/model_comparison.json и reports/feature_analysis.json.",
  "По методичке требуется формирование признаков, отбор и оценка их влияния на качество.",
  "Тестовая permutation importance — постселекционная диагностика стабильности; она не участвовала в выборе модели или признаков.",
  "Слабоструктурированный источник: Банк России RSS https://www.cbr.ru/development/RSS/ .",
];

// Render every slide before export so the updated deck can be visually checked.
for (const [index, page] of presentation.slides.items.entries()) {
  const blob = await presentation.export({ slide: page, format: "png", scale: 1 });
  await fs.writeFile(path.join(slidesPreviewDir, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await blob.arrayBuffer()));
}

await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const { finalizePresentation } = await import(pathToFileURL(
  path.join("C:/Users/skyda/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs"),
).href);
await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath: outputPath,
  pythonExecutable: "C:/Users/skyda/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: "C:/Users/skyda/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/inspect_presentation_package_integrity.py",
  layoutValidatorPath: "C:/Users/skyda/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/inspect_presentation_layout_geometry.py",
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "3", "--require-native-table-slide", "4", "--require-native-table-slide", "5"],
  explicitTotalSlideCount: 11,
  requiredNativeTableOwnerSlides: [3, 4, 5],
  requiredNativeChartOwnerSlides: [6, 7, 9],
  nativeChartTargetApplication: "portable",
  materializeLiteralChartWorkbooks: true,
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "validation-final-v2.json"),
});
console.log(outputPath);
