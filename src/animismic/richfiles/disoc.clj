(comment
  (assoc-in [:on-update-map :update-colors]
    (lib/every-n-seconds
      (let [last-temp (atom 0)]
        (fn [] (lib/normal-distr 0.1 0.1)))
      (let [black? (atom false)]
        (fn [s _]
          (let [color (defs/color-map
                        (if @black?
                          :black
                          (rand-nth [:hit-pink :deep-pink
                                     :green-yellow :white
                                     :cyan])))
                _ (swap! black? not)]
            (lib/update-ents
              s
              (fn [ent]
                (if-not (:update-colors1? ent)
                  ent
                  (if (not @black?)
                    (assoc ent
                      :color (defs/color-map :cyan))
                    (assoc ent :color defs/black)))))))))))
