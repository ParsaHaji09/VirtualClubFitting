import { DEFAULT_LOFT } from './constants';

export const generateFittingRecommendation = (params) => {
  const { clubheadSpeed, attackAngle, spinRate } = params;
  
  let recs = {
    loft: DEFAULT_LOFT,
    shaftFlex: 'Stiff',
    shaftWeight: 65,
    headWeight: 200,
    confidence: 0
  };

  // Logic for Speed and Flex
  if (clubheadSpeed < 85) {
    recs = { ...recs, loft: 12, shaftFlex: 'Senior', shaftWeight: 50 };
  } else if (clubheadSpeed < 95) {
    recs = { ...recs, loft: 10.5, shaftFlex: 'Regular', shaftWeight: 60 };
  } else if (clubheadSpeed < 105) {
    recs = { ...recs, loft: 9.5, shaftFlex: 'Stiff', shaftWeight: 65 };
  } else {
    recs = { ...recs, loft: 9, shaftFlex: 'X-Stiff', shaftWeight: 70 };
  }

  // Adjustment for Attack Angle
  if (attackAngle < -2) recs.loft += 1;
  else if (attackAngle > 3) recs.loft -= 0.5;

  // Spin Optimization Confidence
  recs.headWeight = spinRate > 3000 ? 205 : (spinRate < 2200 ? 195 : 200);
  recs.confidence = spinRate > 3000 || spinRate < 2200 ? 80 : 90;

  return recs;
};