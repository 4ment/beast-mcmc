package dr.app.realtime;

import dr.app.checkpoint.ZippedCheckpointer;
import dr.evolution.tree.BranchRates;
import dr.evolution.tree.NodeRef;
import dr.evomodel.tree.TreeModel;
import dr.evomodel.tree.TreeParameterModel;
import dr.inference.markovchain.MarkovChain;
import dr.inference.model.Model;
import dr.inference.model.Parameter;
import dr.inference.operators.AdaptableMCMCOperator;
import dr.inference.operators.MCMCOperator;
import dr.inference.operators.OperatorSchedule;
import dr.xml.XMLParseException;

import java.io.IOException;
import java.util.*;

/**
 * @author mathieu
 */
public class ZippedCheckpointerModifier extends ZippedCheckpointer {

    private static final boolean DEBUG = false;

    private CheckPointTreeModifier modifyTree;
    private BranchRates rateModel;
    private ArrayList<TreeParameterModel> traitModels;

    public ZippedCheckpointerModifier(String in, String out) throws XMLParseException {
        super(in, out);
    }

    @Override
    public long readNextStateFromZip(MarkovChain markovChain, double[] lnL) {

        OperatorSchedule operatorSchedule = markovChain.getSchedule();
        long state = -1;

        this.traitModels = new ArrayList<TreeParameterModel>();

        StringBuilder ss = new StringBuilder();
        byte[] buffer = new byte[1024];
        int read = 0;

        try {
//            FileReader fileIn = new FileReader(file);
//            BufferedReader in = new BufferedReader(fileIn);
            zipEntry = zin.getNextEntry();
            while ((read =  zin.read(buffer, 0, 1024)) >= 0) {
                ss.append(new String(buffer, 0, read));
            }

            Scanner in = new Scanner(ss.toString());

            int[] rngState = null;

            String line = in.nextLine();
            String[] fields = line.split("\t");
            if (fields[0].equals("rng")) {
                // if there is a random number generator state present then load it...
                try {
                    rngState = new int[fields.length - 1];
                    for (int i = 0; i < rngState.length; i++) {
                        rngState[i] = Integer.parseInt(fields[i + 1]);
                    }

                } catch (NumberFormatException nfe) {
                    throw new RuntimeException("Unable to read state number from state file");
                }

                line = in.nextLine();
                fields = line.split("\t");
            }

            try {
                if (!fields[0].equals("state")) {
                    throw new RuntimeException("Unable to read state number from state file");
                }
                state = Long.parseLong(fields[1]);
            } catch (NumberFormatException nfe) {
                throw new RuntimeException("Unable to read state number from state file");
            }

            line = in.nextLine();
            fields = line.split("\t");

            try {
                if (!fields[0].equals("lnL")) {
                    throw new RuntimeException("Unable to read lnL from state file");
                }
                if (lnL != null) {
                    lnL[0] = Double.parseDouble(fields[1]);
                }
            } catch (NumberFormatException nfe) {
                throw new RuntimeException("Unable to read lnL from state file");
            }

            line = in.nextLine();
            //System.out.println(line);
            fields = line.split("\t");

            //Tree nodes have numbers as parameter ids
            for (Parameter parameter : Parameter.CONNECTED_PARAMETER_SET) {

                //first check if this is actually a tree node by checking if it's a number
                //numbers should be positive but can include zero
                if (isTreeNode(parameter.getId()) && isTreeNode(fields[1]) || parameter.getId().equals(fields[1])) {

                    int dimension = Integer.parseInt(fields[2]);

                    if (dimension != parameter.getDimension() && !fields[1].equals("branchRates.categories")) {
                        System.err.println("Unable to match state parameter dimension: " + dimension + ", expecting " + parameter.getDimension() + " for parameter: " + parameter.getParameterName());
                        System.err.print("Read from file: ");
                        for (int i = 0; i < fields.length; i++) {
                            System.err.print(fields[i] + "\t");
                        }
                        System.err.println();
                    }

                    if (fields[1].equals("branchRates.categories.rootNodeNumber")) {
                        // System.out.println("eek");
                        double value = Double.parseDouble(fields[3]);
                        parameter.setParameterValue(0, value);
                        if (DEBUG) {
                            System.out.println("restoring " + fields[1] + " with value " + value);
                        }
                    } else {
                        if (DEBUG) {
                            System.out.print("restoring " + fields[1] + " with values ");
                        }
                        if (fields[1].equals("branchRates.categories")) {
                            for (int dim = 0; dim < (fields.length-3); dim++) {
                                //System.out.println("dim " + dim);
                                parameter.setParameterValue(dim, Double.parseDouble(fields[dim + 3]));
                                if (DEBUG) {
                                    System.out.print(Double.parseDouble(fields[dim + 3]) + " ");
                                }
                            }
                        } else {
                            for (int dim = 0; dim < parameter.getDimension(); dim++) {
                                parameter.setParameterValue(dim, Double.parseDouble(fields[dim + 3]));
                                if (DEBUG) {
                                    System.out.print(Double.parseDouble(fields[dim + 3]) + " ");
                                }
                            }
                        }
                        if (DEBUG) {
                            System.out.println();
                        }
                    }

                    line = in.nextLine();
                    //System.out.println(line);
                    fields = line.split("\t");

                } else {

                    //there will be more parameters in the connected set than there are lines in the checkpoint file
                    //do nothing and just keep iterating over the parameters in the connected set

                }

            }

            //No changes needed for loading in operators
            for (int i = 0; i < operatorSchedule.getOperatorCount(); i++) {
                MCMCOperator operator = operatorSchedule.getOperator(i);
                if (!fields[1].equals(operator.getOperatorName())) {
                    throw new RuntimeException("Unable to match operator: " + fields[1]);
                }
                if (fields.length < 4) {
                    throw new RuntimeException("Operator missing values: " + fields[1]);
                }
                operator.setAcceptCount(Integer.parseInt(fields[2]));
                operator.setRejectCount(Integer.parseInt(fields[3]));
                if (operator instanceof AdaptableMCMCOperator) {
                    if (fields.length != 5) {
                        throw new RuntimeException("Coercable operator missing parameter: " + fields[1]);
                    }
                    ((AdaptableMCMCOperator)operator).setAdaptableParameter(Double.parseDouble(fields[4]));
                }
                line = in.nextLine();
                fields = line.split("\t");
            }

            // load the tree models last as we get the node heights from the tree (not the parameters which
            // which may not be associated with the right node
            Set<String> expectedTreeModelNames = new HashSet<String>();
            for (Model model : Model.CONNECTED_MODEL_SET) {

                if (model instanceof TreeModel) {
                    expectedTreeModelNames.add(model.getModelName());
                }

                if (model instanceof TreeParameterModel) {
                    this.traitModels.add((TreeParameterModel)model);
                }

                if (model instanceof BranchRates) {
                    this.rateModel = (BranchRates)model;
                }

            }

            while (fields[0].equals("tree")) {

                for (Model model : Model.CONNECTED_MODEL_SET) {
                    if (model instanceof TreeModel && fields[1].equals(model.getModelName())) {

                        //AR: Can we not just add them to a Flexible tree and then make a new TreeModel
                        //taking that in the constructor?

                        //internally, we have a tree with all the taxa
                        //externally, i.e. in the checkpoint file, we have a tree representation comprising
                        //a subset of the full taxa set

                        //write method that adjusts the internal representation, i.e. the one in the connected
                        //set, according to the checkpoint file and a distance-based approach to position
                        //the additional taxa

                        //first read in all the data from the checkpoint file

                        line = in.nextLine();
                        line = in.nextLine();
                        fields = line.split("\t");
                        //read number of nodes
                        int nodeCount = Integer.parseInt(fields[0]);
                        double[] nodeHeights = new double[nodeCount];
                        String[] taxaNames = new String[(nodeCount+1)/2];

                        for (int i = 0; i < nodeCount; i++) {
                            line = in.nextLine();
                            fields = line.split("\t");
                            nodeHeights[i] = Double.parseDouble(fields[1]);
                            if (i < taxaNames.length) {
                                taxaNames[i] = fields[2];
                            }
                        }

                        //on to reading edge information
                        line = in.nextLine();
                        line = in.nextLine();
                        line = in.nextLine();
                        fields = line.split("\t");

                        int edgeCount = Integer.parseInt(fields[0]);

                        //create data matrix of doubles to store information from list of TreeParameterModels
                        double[][] traitValues = new double[traitModels.size()][edgeCount];

                        //create array to store whether a node is left or right child of its parent
                        //can be important for certain tree transition kernels
                        int[] childOrder = new int[edgeCount];
                        for (int i = 0; i < childOrder.length; i++) {
                            childOrder[i] = -1;
                        }

                        int[] parents = new int[edgeCount];
                        for (int i = 0; i < edgeCount; i++){
                            parents[i] = -1;
                        }
                        for (int i = 0; i < edgeCount; i++) {
                            if (in.hasNext()) {
                                line = in.nextLine();
                                fields = line.split("\t");
                                parents[Integer.parseInt(fields[0])] = Integer.parseInt(fields[1]);
                                childOrder[i] = Integer.parseInt(fields[2]);
                                for (int j = 0; j < traitModels.size(); j++) {
                                    traitValues[j][i] = Double.parseDouble(fields[3+j]);
                                }
                            }
                        }

                        //perform magic with the acquired information
                        //CheckPointTreeModifier modifyTree = new CheckPointTreeModifier((TreeModel) model);
                        this.modifyTree = new CheckPointTreeModifier((TreeModel) model);
                        modifyTree.adoptTreeStructure(parents, nodeHeights, childOrder, taxaNames);
                        if (traitModels.size() > 0) {
                            modifyTree.adoptTraitData(parents, this.traitModels, traitValues);
                        }

                        //adopt the loaded tree structure; this does not yet copy the traits on the branches
                        //((TreeModel) model).beginTreeEdit();
                        //((TreeModel) model).adoptTreeStructure(parents, nodeHeights, childOrder);
                        //((TreeModel) model).endTreeEdit();

                        expectedTreeModelNames.remove(model.getModelName());

                    }
                }

                if (in.hasNext()) {
                    fields = in.nextLine().split("\t");
                }

            }

            if (expectedTreeModelNames.size() > 0) {
                StringBuilder sb = new StringBuilder();
                for (String notFoundName : expectedTreeModelNames) {
                    sb.append("Expecting, but unable to match state parameter:" + notFoundName + "\n");
                }
                throw new RuntimeException(sb.toString());
            }

//            in.close();
//            fileIn.close();

        } catch (IOException ioe) {
            throw new RuntimeException("Unable to read file: " + ioe.getMessage());
        }

        return state;
    }

    List<AbstractMap.SimpleEntry<NodeRef, Double>> extendLoadState(CheckPointUpdaterApp.UpdateChoice choice) {
        //add the BranchRates model here
        if (this.rateModel == null) {
            throw new RuntimeException("BranchRates model has not been set correctly.");
        } else {
            List<AbstractMap.SimpleEntry<NodeRef, Double>> results = modifyTree.incorporateAdditionalTaxa(choice, this.rateModel);
            modifyTree.interpolateTraitValues(this.traitModels);
            return results;
        }
    }

    private boolean isTreeNode(String name) {
        if (name == null) {
            return false;
        }
        int length = name.length();
        if (length == 0) {
            return false;
        }
        int i = 0;
        if (name.charAt(0) == '-') {
            return false;
        }
        for (; i < length; i++) {
            char c = name.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }
}
