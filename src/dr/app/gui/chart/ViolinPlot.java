/*
 * NumericalDensityPlot.java
 *
 * Copyright (c) 2002-2015 Alexei Drummond, Andrew Rambaut and Marc Suchard
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

package dr.app.gui.chart;

import dr.inference.trace.TraceDistribution;
import dr.math.distributions.GammaKDEDistribution;
import dr.math.distributions.KernelDensityEstimatorDistribution;
import dr.math.distributions.LogTransformedNormalKDEDistribution;
import dr.math.distributions.NormalKDEDistribution;
import dr.stats.Variate;
import dr.util.FrequencyDistribution;

import java.awt.*;
import java.awt.geom.GeneralPath;
import java.awt.geom.Rectangle2D;
import java.util.*;

/**
 * @author Andrew Rambaut
 */
public class ViolinPlot extends KDENumericalDensityPlot { //Plot.AbstractPlot {
    private final static boolean DEBUG = false;

    private final double violinOffset;
    private final double violinWidth;

    public ViolinPlot(double violinOffset, double violinWidth, java.util.List<Double> data, int minimumBinCount, TraceDistribution traceD) {
        super(data, minimumBinCount, traceD); // TODO Remove when all linked together

//        kde = new GammaKDEDistribution(data);
//
//        System.err.println("Making KDE with " + minimumBinCount + " points");
//
//        Variate xData = getXCoordinates(minimumBinCount);
//        Variate yData = getYCoordinates(xData);
//        setData(xData, yData);
        this.violinOffset = violinOffset;
        this.violinWidth = violinWidth;

        // setData again because in super, the offset and width weren't set.
        setData(data, minimumBinCount);
    }

    /**
     * Set data
     */
    public void setData(Variate.N xData, Variate.N yData) {
        // reverse X and Y because violin plots are rotated
        
        double scale = (0.5 * violinWidth) / ((Number)yData.getMax()).doubleValue();
        this.xData = new Variate.D();
        this.yData = new Variate.D();
        for (int i = 0; i < yData.getCount(); i++) {
            this.xData.add((((Number)yData.get(i)).doubleValue() * scale) + violinOffset);
            this.yData.add((((Number)xData.get(i)).doubleValue()));
        }
        for (int i = yData.getCount() - 1; i >= 0; i--) {
            this.xData.add((-((Number)yData.get(i)).doubleValue() * scale) + violinOffset);
            this.yData.add((((Number)xData.get(i)).doubleValue()));
        }
    }

    /**
     * Paint data series
     */
    protected void paintData(Graphics2D g2, Variate.N xData, Variate.N yData) {

        int n = xData.getCount();

        float x = (float) transformX((((Number)xData.get(0)).doubleValue()));
        float y = (float) transformY(((Number)yData.get(0)).doubleValue());


        GeneralPath path = new GeneralPath();
        path.moveTo(x, y);

        for (int i = 1; i < n; i++) {
            x = (float) transformX((((Number)xData.get(i)).doubleValue()));
            y = (float) transformY(((Number)yData.get(i)).doubleValue());

            path.lineTo(x, y);
        }

        if (solid) {
            path.closePath();
            Paint fillPaint = new Color(
                    ((Color) linePaint).getRed(),
                    ((Color) linePaint).getGreen(),
                    ((Color) linePaint).getBlue(), 32);
            g2.setPaint(fillPaint);
            g2.fill(path);
        }

        g2.setStroke(lineStroke);
        g2.setPaint(linePaint);
        g2.draw(path);
    }


}
