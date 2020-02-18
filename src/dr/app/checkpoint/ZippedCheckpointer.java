package dr.app.checkpoint;

import dr.evolution.tree.NodeRef;
import dr.evomodel.tree.TreeModel;
import dr.evomodel.tree.TreeParameterModel;
import dr.inference.markovchain.MarkovChain;
import dr.inference.model.Likelihood;
import dr.inference.model.Model;
import dr.inference.model.Parameter;
import dr.inference.operators.AdaptableMCMCOperator;
import dr.inference.operators.MCMCOperator;
import dr.inference.operators.OperatorSchedule;
import dr.math.MathUtils;
import dr.xml.XMLParseException;

import java.io.*;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * @author mathieu
 */
public class ZippedCheckpointer {
    protected ZipInputStream zin = null;
    protected ZipOutputStream zout = null;

    public final int size;
    protected ZipEntry zipEntry;
    boolean DEBUG = false;

    public ZippedCheckpointer(String in, String out)throws XMLParseException{
        try {
            zin = new ZipInputStream(new FileInputStream(in));
            zout = new ZipOutputStream(new FileOutputStream(out));
            ZipFile zipFile = new ZipFile(new File(in));
            size = zipFile.size();
            zipFile.close();
        } catch (IOException e) {
            throw new XMLParseException("Cannot read zip file "+ e.getMessage());
        }
    }

    public long readNextStateFromZip(MarkovChain markovChain, double[] lnL) {
        OperatorSchedule operatorSchedule = markovChain.getSchedule();

        long state = -1;

        ArrayList<TreeParameterModel> traitModels = new ArrayList<TreeParameterModel>();

        StringBuilder ss = new StringBuilder();
        byte[] buffer = new byte[1024];
        int read = 0;

        try {
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

            for (Parameter parameter : Parameter.CONNECTED_PARAMETER_SET) {

                line = in.nextLine();
                fields = line.split("\t");
                //if (!fields[0].equals(parameter.getParameterName())) {
                //  System.err.println("Unable to match state parameter: " + fields[0] + ", expecting " + parameter.getParameterName());
                //}
                int dimension = Integer.parseInt(fields[2]);

                if (dimension != parameter.getDimension()) {
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
                    for (int dim = 0; dim < parameter.getDimension(); dim++) {
                        parameter.setParameterValue(dim, Double.parseDouble(fields[dim + 3]));
                        if (DEBUG) {
                            System.out.print(Double.parseDouble(fields[dim + 3]) + " ");
                        }
                    }
                    if (DEBUG) {
                        System.out.println();
                    }
                }

            }

            for (int i = 0; i < operatorSchedule.getOperatorCount(); i++) {
                MCMCOperator operator = operatorSchedule.getOperator(i);
                line = in.nextLine();
                fields = line.split("\t");
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
            }

            // load the tree models last as we get the node heights from the tree (not the parameters which
            // which may not be associated with the right node
            Set<String> expectedTreeModelNames = new HashSet<String>();

            //store list of TreeModels for debugging purposes
            ArrayList<TreeModel> treeModelList = new ArrayList<TreeModel>();

            for (Model model : Model.CONNECTED_MODEL_SET) {

                if (model instanceof TreeModel) {
                    if (DEBUG) {
                        System.out.println("model " + model.getModelName());
                    }
                    treeModelList.add((TreeModel)model);
                    expectedTreeModelNames.add(model.getModelName());
                    if (DEBUG) {
                        System.out.println("\nexpectedTreeModelNames:");
                        for (String s : expectedTreeModelNames) {
                            System.out.println(s);
                        }
                        System.out.println();
                    }
                }

                //first add all TreeParameterModels to a list
                if (model instanceof TreeParameterModel) {
                    traitModels.add((TreeParameterModel)model);
                }

            }

            //explicitly link TreeModel (using its unique ID) to a list of TreeParameterModels
            //this information is currently not yet used
            HashMap<String, ArrayList<TreeParameterModel>> linkedModels = new HashMap<String, ArrayList<TreeParameterModel>>();
            for (String name : expectedTreeModelNames) {
                ArrayList<TreeParameterModel> tpmList = new ArrayList<TreeParameterModel>();
                for (TreeParameterModel tpm : traitModels) {
                    if (tpm.getTreeModel().getId().equals(name)) {
                        tpmList.add(tpm);
                        if (DEBUG) {
                            System.out.println("TreeModel: " + name + " has been assigned TreeParameterModel: " + tpm.toString());
                        }
                    }
                }
                linkedModels.put(name, tpmList);
            }

            line = in.nextLine();
            fields = line.split("\t");
            // Read in all (possibly more than one) trees
            while (fields[0].equals("tree")) {

                if (DEBUG) {
                    System.out.println("\ntree: " + fields[1]);
                }

                for (Model model : Model.CONNECTED_MODEL_SET) {
                    if (model instanceof TreeModel && fields[1].equals(model.getModelName())) {
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
                        if (DEBUG) {
                            System.out.println("edge count = " + edgeCount);
                        }

                        //create data matrix of doubles to store information from list of TreeParameterModels
                        //size of matrix depends on the number of TreeParameterModels assigned to a TreeModel
                        double[][] traitValues = new double[linkedModels.get(model.getId()).size()][edgeCount];

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
                        for (int i = 0; i < edgeCount-1; i++) {
                            if (in.hasNext()) {
                                line = in.nextLine();
                                if (DEBUG) {
                                    System.out.println("DEBUG: " + line);
                                }
                                fields = line.split("\t");
                                parents[Integer.parseInt(fields[0])] = Integer.parseInt(fields[1]);
                                // childOrder[i] = Integer.parseInt(fields[2]);
                                childOrder[Integer.parseInt(fields[0])] = Integer.parseInt(fields[2]);
                                for (int j = 0; j < linkedModels.get(model.getId()).size(); j++) {
                                    //   traitValues[j][i] = Double.parseDouble(fields[3+j]);
                                    traitValues[j][Integer.parseInt(fields[0])] = Double.parseDouble(fields[3+j]);
                                }
                            }
                        }

                        //perform magic with the acquired information
                        if (DEBUG) {
                            System.out.println("adopting tree structure");
                        }

                        //adopt the loaded tree structure;
                        ((TreeModel) model).beginTreeEdit();
                        ((TreeModel) model).adoptTreeStructure(parents, nodeHeights, childOrder, taxaNames);
                        if (traitModels.size() > 0) {
                            ((TreeModel) model).adoptTraitData(parents, traitModels, traitValues, taxaNames);
                        }
                        ((TreeModel) model).endTreeEdit();

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
                throw new RuntimeException("\n" + sb.toString());
            }

            if (DEBUG) {
                System.out.println("\nDouble checking:");
                for (Parameter parameter : Parameter.CONNECTED_PARAMETER_SET) {
                    if (parameter.getParameterName().equals("branchRates.categories.rootNodeNumber")) {
                        System.out.println(parameter.getParameterName() + ": " + parameter.getParameterValue(0));
                    }
                }
                System.out.println("\nPrinting trees:");
                for (TreeModel tm : treeModelList) {
                    System.out.println(tm.getId() + ": ");
                    System.out.println(tm.getNewick());
                }
            }

            if (rngState != null) {
                MathUtils.setRandomState(rngState);
            }

            in.close();
//            fileIn.close();

            //This shouldn't be necessary and if it is then it might be hiding a bug...
            /*for (Likelihood likelihood : Likelihood.CONNECTED_LIKELIHOOD_SET) {
                likelihood.makeDirty();
            }*/

        } catch (IOException ioe) {
            throw new RuntimeException("Unable to read file: " + ioe.getMessage());
        }

        return state;
    }

    public boolean writeStateToZip(long state, double lnL, MarkovChain markovChain, List<AbstractMap.SimpleEntry<NodeRef, Double>> probs) {
        OperatorSchedule operatorSchedule = markovChain.getSchedule();
        String currentStateFile = zipEntry.getName();

        ZipEntry ze = new ZipEntry(currentStateFile);


//        OutputStream fileOut = null;
        try {
//            fileOut = new FileOutputStream(file);
            zout.putNextEntry(ze);
            PrintStream out = new PrintStream(zout);

            ArrayList<TreeParameterModel> traitModels = new ArrayList<TreeParameterModel>();

            int[] rngState = MathUtils.getRandomState();
            out.print("rng");
            for (int i = 0; i < rngState.length; i++) {
                out.print("\t");
                out.print(rngState[i]);
            }
            out.println();

            out.print("state\t");
            out.println(state);

            out.print("lnL\t");
            out.print(lnL);
            for (AbstractMap.SimpleEntry pair : probs){
                out.print("\t" + pair.getValue());
            }
            out.println();

            for (Parameter parameter : Parameter.CONNECTED_PARAMETER_SET) {
                if (!parameter.isImmutable()) {
                    out.print("parameter");
                    out.print("\t");
                    out.print(parameter.getParameterName());
                    out.print("\t");
                    out.print(parameter.getDimension());
                    for (int dim = 0; dim < parameter.getDimension(); dim++) {
                        out.print("\t");
                        out.print(parameter.getParameterValue(dim));
                    }
                    out.println();
                }
            }

            for (int i = 0; i < operatorSchedule.getOperatorCount(); i++) {
                MCMCOperator operator = operatorSchedule.getOperator(i);
                out.print("operator");
                out.print("\t");
                out.print(operator.getOperatorName());
                out.print("\t");
                out.print(operator.getAcceptCount());
                out.print("\t");
                out.print(operator.getRejectCount());
                if (operator instanceof AdaptableMCMCOperator) {
                    out.print("\t");
                    out.print(((AdaptableMCMCOperator)operator).getAdaptableParameter());
                }
                out.println();
            }

            //check up front if there are any TreeParameterModel objects
            for (Model model : Model.CONNECTED_MODEL_SET) {
                if (model instanceof TreeParameterModel) {
                    //System.out.println("\nDetected TreeParameterModel: " + ((TreeParameterModel) model).toString());
                    traitModels.add((TreeParameterModel) model);
                }
            }

            for (Model model : Model.CONNECTED_MODEL_SET) {

                if (model instanceof TreeModel) {
                    out.print("tree");
                    out.print("\t");
                    out.println(model.getModelName());

                    //replace Newick format by printing general graph structure
                    //out.println(((TreeModel) model).getNewick());

                    out.println("#node height taxon");
                    int nodeCount = ((TreeModel) model).getNodeCount();
                    out.println(nodeCount);
                    for (int i = 0; i < nodeCount; i++) {
                        out.print(((TreeModel) model).getNode(i).getNumber());
                        out.print("\t");
                        out.print(((TreeModel) model).getNodeHeight(((TreeModel) model).getNode(i)));
                        if (((TreeModel) model).isExternal(((TreeModel) model).getNode(i))) {
                            out.print("\t");
                            out.print(((TreeModel) model).getNodeTaxon(((TreeModel) model).getNode(i)).getId());
                        }
                        out.println();
                    }

                    out.println("#edges");
                    out.println("#child-node parent-node L/R-child traits");

                    out.println(nodeCount);
                    for (int i = 0; i < nodeCount; i++) {
                        NodeRef parent = ((TreeModel) model).getParent(((TreeModel) model).getNode(i));
                        if (parent != null) {
                            out.print(((TreeModel) model).getNode(i).getNumber());
                            out.print("\t");
                            out.print(((TreeModel) model).getParent(((TreeModel) model).getNode(i)).getNumber());
                            out.print("\t");

                            if ((((TreeModel) model).getChild(parent, 0) == ((TreeModel) model).getNode(i))) {
                                //left child
                                out.print(0);
                            } else if ((((TreeModel) model).getChild(parent, 1) == ((TreeModel) model).getNode(i))) {
                                //right child
                                out.print(1);
                            } else {
                                throw new RuntimeException("Operation currently only supported for nodes with 2 children.");
                            }

                            //only print the TreeParameterModel that matches the TreeModel currently being written
                            for (TreeParameterModel tpm : traitModels) {
                                if (model == tpm.getTreeModel()) {
                                    out.print("\t");
                                    out.print(tpm.getNodeValue((TreeModel) model, ((TreeModel) model).getNode(i)));
                                }
                            }
                            out.println();
                        } else {
                            if (DEBUG) {
                                System.out.println(((TreeModel) model).getNode(i) + " has no parent.");
                            }
                        }
                    }

                }

            }
//            out.close();
//            fileOut.close();
        } catch (IOException ioe) {
            System.err.println("Unable to write file: " + ioe.getMessage());
            return false;
        }

        if (DEBUG) {
            for (Likelihood likelihood : Likelihood.CONNECTED_LIKELIHOOD_SET) {
                System.err.println(likelihood.getId() + ": " + likelihood.getLogLikelihood());
            }
        }

        return true;
    }

    public void close(){
        try {
            zin.close();
            zout.close();
        } catch (IOException e) {
            System.err.println("Unable to close zip file: " + e.getMessage());
        }
    }
}
