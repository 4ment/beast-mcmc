/*
 * TreeTipGradient.java
 *
 * Copyright (c) 2002-2017 Alexei Drummond, Andrew Rambaut and Marc Suchard
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

package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.tree.TreeTrait;
import dr.evomodel.branchratemodel.ArbitraryBranchRates;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.treedatalikelihood.TreeDataLikelihood;
import dr.evomodel.treedatalikelihood.continuous.cdi.SafeMultivariateWithDriftIntegrator;
import dr.evomodel.treedatalikelihood.preorder.BranchConditionalDistributionDelegate;
import dr.evomodel.treedatalikelihood.preorder.BranchSufficientStatistics;
import dr.evomodel.treedatalikelihood.preorder.NormalSufficientStatistics;
import dr.inference.hmc.GradientWrtParameterProvider;
import dr.inference.model.Likelihood;
import dr.inference.model.Parameter;
import dr.math.MultivariateFunction;
import dr.math.NumericalDerivative;
import dr.math.matrixAlgebra.WrappedVector;
import dr.xml.Reportable;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.List;

import static dr.math.matrixAlgebra.missingData.MissingOps.safeInvert;
import static dr.math.matrixAlgebra.missingData.MissingOps.safeWeightedAverage;

/**
 * @author Marc A. Suchard
 */
public class BranchRateGradient implements GradientWrtParameterProvider, Reportable {

    private final TreeDataLikelihood treeDataLikelihood;
    private final TreeTrait<List<BranchSufficientStatistics>> treeTraitProvider;
    private final Tree tree;
    private final int nTraits;
    private final int dim;
    private final Parameter rateParameter;
    private final ArbitraryBranchRates branchRateModel;
//    private final MatrixParameterInterface diffusionParameter;

//    private final DenseMatrix64F L;
//    private final DenseMatrix64F Lt;
    private final DenseMatrix64F matrix0;
    private final DenseMatrix64F matrix1;
//    private final DenseMatrix64F matrix2;
    private final DenseMatrix64F vector0;

    public BranchRateGradient(String traitName,
                              TreeDataLikelihood treeDataLikelihood,
                              ContinuousDataLikelihoodDelegate likelihoodDelegate,
                              Parameter rateParameter) {

        assert(treeDataLikelihood != null);

        this.treeDataLikelihood = treeDataLikelihood;
        this.tree = treeDataLikelihood.getTree();
        this.rateParameter = rateParameter;
//        this.diffusionParameter = likelihoodDelegate.getDiffusionModel().getPrecisionParameter();

        BranchRateModel brm = treeDataLikelihood.getBranchRateModel();
        this.branchRateModel = (brm instanceof ArbitraryBranchRates) ? (ArbitraryBranchRates) brm : null;

        // TODO Move into different constructor / parser
        String bcdName = BranchConditionalDistributionDelegate.getName(traitName);
        if (treeDataLikelihood.getTreeTrait(bcdName) == null) {
            likelihoodDelegate.addBranchConditionalDensityTrait(traitName);
        }

        @SuppressWarnings("unchecked")
        TreeTrait<List<BranchSufficientStatistics>> unchecked = treeDataLikelihood.getTreeTrait(bcdName);
        treeTraitProvider = unchecked;

        assert (treeTraitProvider != null);

        nTraits = treeDataLikelihood.getDataLikelihoodDelegate().getTraitCount();
        if (nTraits != 1) {
            throw new RuntimeException("Not yet implemented for >1 traits");
        }
        dim = treeDataLikelihood.getDataLikelihoodDelegate().getTraitDim();
//        assert (dim == diffusionParameter.getColumnDimension());

        if (likelihoodDelegate.getIntegrator() instanceof SafeMultivariateWithDriftIntegrator) {
            throw new RuntimeException("Not yet implemented with drift");
        }

//        L = new DenseMatrix64F(dim, dim);
//        Lt = new DenseMatrix64F(dim, dim);
        matrix0 = new DenseMatrix64F(dim, dim);
        matrix1 = new DenseMatrix64F(dim, dim);
//        matrix2 = new DenseMatrix64F(dim, dim);
        vector0 = new DenseMatrix64F(dim, 1);
    }

    @Override
    public Likelihood getLikelihood() {
        return treeDataLikelihood;
    }

    @Override
    public Parameter getParameter() {
        return rateParameter;
    }

    @Override
    public int getDimension() {
        return getParameter().getDimension();
    }

    @Override
    public double[] getGradientLogDensity() {

        double[] result = new double[rateParameter.getDimension()];

        // TODO Do single call to traitProvider with node == null (get full tree)
//        List<BranchSufficientStatistics> statisticsForTree = (List<BranchSufficientStatistics>)
//                treeTraitProvider.getTrait(tree, null);

        for (int i = 0; i < tree.getNodeCount(); ++i) {
            final NodeRef node = tree.getNode(i);

            if (!tree.isRoot(node)) {

                List<BranchSufficientStatistics> statisticsForNode = treeTraitProvider.getTrait(tree, node);

                assert (statisticsForNode.size() == nTraits);

                double differential = branchRateModel.getBranchRateDifferential(tree, node);

                double gradient = 0.0;
                for (int trait = 0; trait < nTraits; ++trait) {
                    gradient += getGradientForBranch(statisticsForNode.get(trait), differential);
                }

                final int destinationIndex = getParameterIndexFromNode(node);
                assert (destinationIndex != -1);

                result[destinationIndex] = gradient;
            }
        }

        return result;
    }

//    private void getCholeskyDecomposition(MatrixParameterInterface input, DenseMatrix64F L, DenseMatrix64F Lt) {
//
//        CholeskyDecomposition<DenseMatrix64F> cholesky = DecompositionFactory.chol(dim, true);
//        DenseMatrix64F matrix = DenseMatrix64F.wrap(dim, dim, input.getParameterValues());
//
//        if( !cholesky.decompose(matrix)) {
//            throw new RuntimeException("Cholesky decomposition failed!");
//        }
//
////        return cholesky.getT(null);
//        cholesky.getT(L);
//        CommonOps.transpose(L, Lt);
//    }
//
//    private void getCholeskyDecomposition(DenseMatrix64F input, DenseMatrix64F L, DenseMatrix64F Lt) {
//
//        CholeskyDecomposition<DenseMatrix64F> cholesky = DecompositionFactory.chol(dim, true);
//
//        if( !cholesky.decompose(input)) {
//            throw new RuntimeException("Cholesky decomposition failed!");
//        }
//
////        return cholesky.getT(null);
//        cholesky.getT(L);
//        CommonOps.transpose(L, Lt);
//    }
//

    private int getParameterIndexFromNode(NodeRef node) {
        return (branchRateModel == null) ? node.getNumber() : branchRateModel.getParameterIndexFromNode(node);
    }


    private NormalSufficientStatistics computeJointStatistics(NormalSufficientStatistics child,
                                                              NormalSufficientStatistics parent) {

        @SuppressWarnings("deprecated")
        DenseMatrix64F totalP = new DenseMatrix64F(dim, dim);
        CommonOps.add(child.getRawPrecision(), parent.getRawPrecision(), totalP);

        DenseMatrix64F totalV = new DenseMatrix64F(dim, dim);
        safeInvert(totalP, totalV, false);

        DenseMatrix64F mean = new DenseMatrix64F(dim, 1);
        safeWeightedAverage(
                new WrappedVector.Raw(child.getRawMean().getData(), 0, dim),
                child.getRawPrecision(),
                new WrappedVector.Raw(parent.getRawMean().getData(), 0, dim),
                parent.getRawPrecision(),
                new WrappedVector.Raw(mean.getData(), 0, dim),
                totalV,
                dim);

        return new NormalSufficientStatistics(mean, totalP, totalV);
    }

    private double getGradientForBranch(BranchSufficientStatistics statistics, double differentialScaling) {

        final NormalSufficientStatistics child = statistics.getChild();
        final NormalSufficientStatistics branch = statistics.getBranch();
        final NormalSufficientStatistics parent = statistics.getParent();

        if (DEBUG) {
//            System.err.println("L = " + NormalSufficientStatistics.toVectorizedString(L));
            System.err.println("B = " + statistics.toVectorizedString());
        }

        DenseMatrix64F Qi = parent.getRawPrecision();
        DenseMatrix64F Si = matrix1; //new DenseMatrix64F(dim, dim); // matrix1;
        CommonOps.scale(differentialScaling, branch.getRawVariance(), Si);

        if (DEBUG) {
            System.err.println("\tQi = " + NormalSufficientStatistics.toVectorizedString(Qi));
            System.err.println("\tSi = " + NormalSufficientStatistics.toVectorizedString(Si));
        }

        DenseMatrix64F logDetComponent = matrix0; //new DenseMatrix64F(dim, dim); //matrix0;
        CommonOps.mult(Qi, Si, logDetComponent);

        double grad1 = 0.0;
        for (int row = 0; row < dim; ++row) {
            grad1 -= 0.5 * logDetComponent.unsafe_get(row, row);
        }

        if (DEBUG) {
            System.err.println("grad1 = " + grad1);
        }

        DenseMatrix64F quadraticComponent = matrix1; //new DenseMatrix64F(dim, dim); //matrix1;
        CommonOps.mult(logDetComponent, Qi, quadraticComponent);

        if (DEBUG) {
            System.err.println("\tQi = " + NormalSufficientStatistics.toVectorizedString(Qi));
            System.err.println("\tQQ = " + NormalSufficientStatistics.toVectorizedString(quadraticComponent));
        }

        NormalSufficientStatistics jointStatistics = computeJointStatistics(child, parent);

        DenseMatrix64F additionalVariance = matrix0; //new DenseMatrix64F(dim, dim);
        CommonOps.mult(quadraticComponent, jointStatistics.getRawVariance(), additionalVariance);

        DenseMatrix64F delta = vector0;
        for (int row = 0; row < dim; ++row) {
            delta.unsafe_set(row, 0,
                    jointStatistics.getRawMean().unsafe_get(row, 0) - parent.getMean(row) // TODO Correct for drift
            );
        }

        double grad2 = 0.0;
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {

                grad2 += 0.5 * delta.unsafe_get(row, 0)
                        * quadraticComponent.unsafe_get(row, col)
                        * delta.unsafe_get(col, 0);

            }

            grad2 += 0.5 * additionalVariance.unsafe_get(row, row);
        }

        if (DEBUG) {
            System.err.println("\tchild variance = " + NormalSufficientStatistics.toVectorizedString(child.getRawVariance()));
            System.err.println("\tquadratic      = " + NormalSufficientStatistics.toVectorizedString(quadraticComponent));
            System.err.println("\tadditional     = " + NormalSufficientStatistics.toVectorizedString(additionalVariance));
            System.err.println("delta: " + NormalSufficientStatistics.toVectorizedString(delta));
            System.err.println("grad2 = " + grad2);
        }

        return grad1 + grad2;
    }



    private MultivariateFunction numeric1 = new MultivariateFunction() {
        @Override
        public double evaluate(double[] argument) {

            for (int i = 0; i < argument.length; ++i) {
                rateParameter.setParameterValue(i, argument[i]);
            }

            treeDataLikelihood.makeDirty();
            return treeDataLikelihood.getLogLikelihood();
        }

        @Override
        public int getNumArguments() {
            return rateParameter.getDimension();
        }

        @Override
        public double getLowerBound(int n) {
            return 0;
        }

        @Override
        public double getUpperBound(int n) {
            return Double.POSITIVE_INFINITY;
        }
    };

    @Override
    public String getReport() {

        double[] savedValues = rateParameter.getParameterValues();
        double[] testGradient = NumericalDerivative.gradient(numeric1, rateParameter.getParameterValues());
        for (int i = 0; i < savedValues.length; ++i) {
            rateParameter.setParameterValue(i, savedValues[i]);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Peeling: ").append(new dr.math.matrixAlgebra.Vector(getGradientLogDensity()));
        sb.append("\n");
        sb.append("numeric: ").append(new dr.math.matrixAlgebra.Vector(testGradient));
        sb.append("\n");
        
        return sb.toString();
    }

    private static final boolean DEBUG = false;
}
