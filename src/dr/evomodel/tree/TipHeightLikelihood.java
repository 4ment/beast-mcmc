/*
 * TipHeightLikelihood.java
 *
 * Copyright (C) 2002-2006 Alexei Drummond and Andrew Rambaut
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

package dr.evomodel.tree;

import dr.inference.distribution.ParametricDistributionModel;
import dr.inference.model.Likelihood;
import dr.inference.model.Parameter;
import dr.xml.*;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 * A class that returns the log likelihood of a set of data (statistics)
 * being distributed according to the given parametric distribution.
 * @author Alexei Drummond
 * @version $Id: TipHeightLikelihood.java,v 1.2 2005/05/17 16:54:41 rambaut Exp $
 */

public class TipHeightLikelihood extends Likelihood.Abstract {

	public static final String TIP_HEIGHT_LIKELIHOOD = "tipHeightLikelihood";

	public static final String DISTRIBUTION = "distribution";
	public static final String TIP_HEIGHTS = "tipHeights";

	public TipHeightLikelihood(ParametricDistributionModel distribution, Parameter tipHeights) {
        super(distribution);
		this.distribution = distribution;
		this.tipHeights = tipHeights;
		offsets = new double[tipHeights.getDimension()];
		for (int i = 0; i < tipHeights.getDimension(); i++) {
			offsets[i] = tipHeights.getParameterValue(i);
		}
	}

	// **************************************************************
    // Likelihood IMPLEMENTATION
    // **************************************************************

	/**
     * Calculate the log likelihood of the current state.
     * @return the log likelihood.
     */
	public double calculateLogLikelihood() {

		double logL = 0.0;

		for (int i = 0; i < tipHeights.getDimension(); i++) {
			logL += distribution.logPdf(tipHeights.getParameterValue(i) - offsets[i]);
		}

		return logL;
	}


	/**
	 * Overridden to always return false.
	 */
	protected boolean getLikelihoodKnown() {
		return false;
	}

	// **************************************************************
    // XMLElement IMPLEMENTATION
    // **************************************************************

	public Element createElement(Document d) {
		throw new RuntimeException("Not implemented yet!");
	}


	/**
	 * Reads a distribution likelihood from a DOM Document element.
	 */
	public static XMLObjectParser PARSER = new AbstractXMLObjectParser() {

		public String getParserName() { return TIP_HEIGHT_LIKELIHOOD; }

		public Object parseXMLObject(XMLObject xo) throws XMLParseException {

			ParametricDistributionModel model = (ParametricDistributionModel)xo.getElementFirstChild(DISTRIBUTION);
			Parameter tipHeights = (Parameter)xo.getElementFirstChild(TIP_HEIGHTS);

			return new TipHeightLikelihood(model, tipHeights);
		}

		//************************************************************************
		// AbstractXMLObjectParser implementation
		//************************************************************************

		public XMLSyntaxRule[] getSyntaxRules() { return rules; }

		private XMLSyntaxRule[] rules = new XMLSyntaxRule[] {
			new ElementRule(DISTRIBUTION,
				new XMLSyntaxRule[] { new ElementRule(ParametricDistributionModel.class) }),
			new ElementRule(TIP_HEIGHTS,
				new XMLSyntaxRule[] { new ElementRule(Parameter.class) }),
		};

		public String getParserDescription() {
			return "Calculates the likelihood of the tipHeights given some parametric or empirical distribution.";
		}

		public Class getReturnType() { return Likelihood.class; }
	};

	ParametricDistributionModel distribution;
	private final Parameter tipHeights;
	private final double[] offsets;
}

