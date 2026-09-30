const SAMPLE_ROWS = 9000;

const seededRandom = (seed) => () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const csvValue = (value) => {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

const makeCsvFile = (filename, headers, rows) => {
  const csv = [headers, ...rows]
    .map((row) => row.map(csvValue).join(","))
    .join("\n");
  return new File([csv], filename, { type: "text/csv" });
};

const buildChurnSample = () => {
  const random = seededRandom(202503);
  const plans = ["basic", "plus", "premium"];
  const regions = ["north", "south", "east", "west"];
  const rows = Array.from({ length: SAMPLE_ROWS }, () => {
    const age = 18 + Math.floor(random() * 63);
    const monthsActive = 1 + Math.floor(random() * 72);
    const monthlySpend = Number((18 + random() * 182).toFixed(2));
    const supportTickets = Math.floor(random() * 7);
    const plan = plans[Math.floor(random() * plans.length)];
    const region = regions[Math.floor(random() * regions.length)];
    const autoPay = random() < 0.68 ? "yes" : "no";
    const risk = -2.1 + supportTickets * 0.55 + (autoPay === "no" ? 0.8 : 0)
      + (plan === "basic" ? 0.65 : plan === "premium" ? -0.45 : 0)
      - monthsActive * 0.025 + (monthlySpend > 145 ? 0.2 : 0);
    const churnChance = 1 / (1 + Math.exp(-risk));
    const churned = random() < churnChance ? "yes" : "no";
    return [age, monthsActive, monthlySpend, supportTickets, plan, region, autoPay, churned];
  });

  return makeCsvFile("smartml_customer_churn_sample.csv", [
    "customer_age", "months_active", "monthly_spend", "support_tickets",
    "plan", "region", "auto_pay", "churned",
  ], rows);
};

const buildRentSample = () => {
  const random = seededRandom(202506);
  const cities = ["Austin", "Denver", "Chicago", "Atlanta", "Portland"];
  const propertyTypes = ["apartment", "condo", "townhouse", "studio"];
  const rows = Array.from({ length: SAMPLE_ROWS }, () => {
    const bedrooms = Math.floor(random() * 5);
    const bathrooms = Number((1 + Math.floor(random() * 3) * 0.5).toFixed(1));
    const areaSqft = Math.round(450 + random() * 2350);
    const ageYears = Math.floor(random() * 55);
    const distanceToCenter = Number((0.5 + random() * 24).toFixed(1));
    const city = cities[Math.floor(random() * cities.length)];
    const propertyType = propertyTypes[Math.floor(random() * propertyTypes.length)];
    const cityEffect = { Austin: 210, Denver: 250, Chicago: 95, Atlanta: 75, Portland: 185 }[city];
    const typeEffect = { apartment: 80, condo: 145, townhouse: 230, studio: -120 }[propertyType];
    const monthlyRent = Math.max(450, Math.round(
      530 + areaSqft * 1.08 + bedrooms * 175 + bathrooms * 115 + cityEffect
      + typeEffect - ageYears * 5.5 - distanceToCenter * 18 + (random() - 0.5) * 340
    ));
    return [bedrooms, bathrooms, areaSqft, ageYears, distanceToCenter, city, propertyType, monthlyRent];
  });

  return makeCsvFile("smartml_rent_prediction_sample.csv", [
    "bedrooms", "bathrooms", "area_sqft", "property_age_years",
    "distance_to_center_km", "city", "property_type", "monthly_rent",
  ], rows);
};

export const SAMPLE_DATASETS = {
  classification: {
    title: "Customer churn",
    description: "Predict whether a customer is likely to leave.",
    taskType: "classification",
    target: "churned",
    columns: ["customer_age", "months_active", "monthly_spend", "support_tickets", "plan", "region", "auto_pay", "churned"],
    makeFile: buildChurnSample,
  },
  regression: {
    title: "Rental prices",
    description: "Estimate monthly rent from property details.",
    taskType: "regression",
    target: "monthly_rent",
    columns: ["bedrooms", "bathrooms", "area_sqft", "property_age_years", "distance_to_center_km", "city", "property_type", "monthly_rent"],
    makeFile: buildRentSample,
  },
};

export const SAMPLE_ROW_COUNT = SAMPLE_ROWS;
