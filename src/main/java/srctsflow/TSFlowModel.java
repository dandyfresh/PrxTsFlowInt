/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package srctsflow;

import com.src.opencv.md.dto.item;
import com.src.opencv.md.intf.ModModel;
import java.util.List;
import org.tensorflow.SavedModelBundle;

/**
 *
 * @author cursi
 */
public class TSFlowModel implements ModModel{
     private double minScore;
    /**
     * @return the minScore
     */
    public double getMinScore() {
        return minScore;
    }

    /**
     * @param minScore the minScore to set
     */
    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }
    /**
     * @return the configModelPath
     */
    @Override
    public String getConfigModelPath() {
        return configModelPath;
    }

    /**
     * @param configModelPath the configModelPath to set
     */
    @Override
    public void setConfigModelPath(String configModelPath) {
        this.configModelPath = configModelPath;
    }

    /**
     * @return the weightModel
     */
    @Override
    public Object getWeightModel() {
        return weightModel;
    }

    /**
     * @param weightModel the weightModel to set
     */
    @Override
    public void setWeightModel(Object weightModel) {
        this.weightModel = (SavedModelBundle)weightModel;
    }

    /**
     * @return the items
     */
    @Override
    public List<item> getItems() {
        return items;
    }

    /**
     * @param items the items to set
     */
    public void setItems(List<item> items) {
        this.items = items;
    }
    private String configModelPath;
    private SavedModelBundle weightModel;
    private List<item> items;


    public String getItemName(int id) {
        String res=null;
        if (items != null) {
            for (int i = 0; i < this.items.size(); i++) {
                if(items.get(i).getId()==id){
                    return items.get(i).getName();
                }
            }
        }
        return res;
    }

    @Override
    public String getOpts() {
        return opts;
    }

    @Override
    public void setOpts(String opts) {
        this.opts=opts;
    }
    String opts;
     /**
     * @return the clase
     */
    @Override
    public String getClase() {
        return clase;
    }

    /**
     * @param clase the clase to set
     */
    @Override
    public void setClase(String clase) {
        this.clase = clase;
    }
    private String clase;

}
