    def forward(self, x):
        '''
        :param x: [b, 3, 32, 32]
        :return
        '''
        batchsz = x.size(0)
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)
        #[b, 16, 5, 5] => [b, 16 * 5 * 5]
        logits = self.fc_unit(x)
        
        #pred = F.softmax(logits, dim=1)
        #loss = self.criteon(logits, y)

        return logits